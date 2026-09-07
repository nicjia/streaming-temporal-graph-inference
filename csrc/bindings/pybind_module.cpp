#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/pcsr_graph.hpp"
#include "../include/streaming_ingestor.hpp"

namespace py = pybind11;

namespace {

/**
 * Strip NPY_ARRAY_WRITEABLE from a view that aliases C++-owned memory.
 *
 * These arrays point straight at the arena. Letting Python write through them
 * would let a stray assignment plant an EMPTY_GAP in the middle of a run or
 * push a target id past num_vertices, corrupting the graph with no way to
 * detect it. Readers get the zero-copy speed; mutation goes through insert_edge.
 */
template <typename T>
py::array_t<T> as_readonly(py::array_t<T> arr) {
    py::detail::array_proxy(arr.ptr())->flags &=
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return arr;
}

} // namespace

PYBIND11_MODULE(graph_engine, m) {
    m.doc() = "Zero-allocation High-Throughput Graph Engine";

    py::class_<PCSRGraph>(m, "PCSRGraph")
        .def(py::init<uint32_t, uint32_t, size_t, bool>(),
             py::arg("max_vertices"),
             py::arg("initial_edge_capacity"),
             py::arg("arena_bytes") = 128 * 1024 * 1024,
             py::arg("store_weights") = false)

        .def("insert_edge", [](PCSRGraph& self, uint32_t src, uint32_t dst,
                               uint32_t timestamp, EdgeRelation relation,
                               float weight) {
            // Guarded too: the scalar path holds the GIL, but a bulk insert on
            // another thread has released it, so the two can still overlap.
            PCSRGraph::WriteGuard guard(self);
            self.insert_edge(src, dst, timestamp, relation, weight);
        }, py::arg("src"), py::arg("dst"), py::arg("timestamp"),
           py::arg("relation") = RELATION_UNKNOWN, py::arg("weight") = 0.0f)

        .def("expire_before", [](PCSRGraph& self, uint32_t timestamp) {
            PCSRGraph::WriteGuard guard(self);
            return self.expire_before(timestamp);
        }, py::arg("timestamp"),
           "Drop the leading run of every adjacency older than `timestamp` and "
           "return how many edges went. O(V + edges removed), no memory movement: "
           "the freed slots are reclaimed by the next rebalance of their window. "
           "Invalidates sampler views -- call PCSRTemporalSampler.refresh().")

        .def("expire_vertex_before", [](PCSRGraph& self, uint32_t vertex,
                                        uint32_t timestamp) {
            PCSRGraph::WriteGuard guard(self);
            return self.expire_vertex_before(vertex, timestamp);
        }, py::arg("vertex"), py::arg("timestamp"),
           "expire_before() restricted to one vertex.")

        .def("insert_edges", [](PCSRGraph& self,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> src,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dst,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ts,
                                py::object relations,
                                py::object weights) {
            // Bulk path. Calling insert_edge() once per event costs a pybind
            // dispatch plus a Python int box per endpoint -- microseconds --
            // which swamps the ~15 ns the engine actually spends. Handing over
            // three numpy buffers crosses the boundary once for the whole batch
            // and drops the GIL for the duration, which is the entire point of
            // decoupling the ingestion thread from the engine.
            if (src.ndim() != 1 || dst.ndim() != 1 || ts.ndim() != 1) {
                throw std::invalid_argument("insert_edges expects 1-D arrays");
            }
            if (src.size() != dst.size() || src.size() != ts.size()) {
                throw std::invalid_argument("src, dst and timestamp arrays must be the same length");
            }

            const uint32_t* sp = src.data();
            const uint32_t* dp = dst.data();
            const uint32_t* tp = ts.data();
            const size_t n = static_cast<size_t>(src.size());

            const EdgeRelation* rp = nullptr;
            py::array_t<EdgeRelation, py::array::c_style | py::array::forcecast> rel_array;
            if (!relations.is_none()) {
                rel_array = relations.cast<py::array_t<EdgeRelation,
                                py::array::c_style | py::array::forcecast>>();
                if (static_cast<size_t>(rel_array.size()) != n) {
                    throw std::invalid_argument(
                        "relation array must match the edge arrays in length");
                }
                rp = rel_array.data();
            }

            const float* wp = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> weight_array;
            if (!weights.is_none()) {
                weight_array = weights.cast<py::array_t<float,
                                   py::array::c_style | py::array::forcecast>>();
                if (static_cast<size_t>(weight_array.size()) != n) {
                    throw std::invalid_argument(
                        "weight array must match the edge arrays in length");
                }
                wp = weight_array.data();
            }

            // Acquired before the GIL is dropped, so a rejection surfaces as
            // a normal Python exception rather than unwinding through a
            // GIL-released region.
            PCSRGraph::WriteGuard guard(self);

            py::gil_scoped_release release;
            for (size_t i = 0; i < n; ++i) {
                self.insert_edge(sp[i], dp[i], tp[i],
                                 rp ? rp[i] : RELATION_UNKNOWN,
                                 wp ? wp[i] : 0.0f);
            }
            return n;
        }, py::arg("src"), py::arg("dst"), py::arg("timestamp"),
           py::arg("relation") = py::none(), py::arg("weight") = py::none(),
           "Insert a whole batch from numpy arrays in one crossing, with the "
           "GIL released. Returns the number of edges inserted.")

        .def("get_vertex_offsets", [](PCSRGraph& self) {
            uint32_t* data_ptr = const_cast<uint32_t*>(self.get_vertex_offsets());
            uint32_t size = self.get_num_vertices() + 1;

            return as_readonly(py::array_t<uint32_t>(
                {size},
                {sizeof(uint32_t)},
                data_ptr,
                py::cast(self)
            ));
        }, "Zero-copy view of the region boundaries (length V + 1).")

        .def("get_edge_relations", [](PCSRGraph& self) {
            EdgeRelation* data_ptr = const_cast<EdgeRelation*>(self.get_edge_relations());
            uint32_t size = self.get_edge_capacity();

            return as_readonly(py::array_t<EdgeRelation>(
                {size},
                {sizeof(EdgeRelation)},
                data_ptr,
                py::cast(self)
            ));
        }, "Zero-copy view of the per-slot relation type, parallel to get_edges().")

        .def("get_edge_weights", [](PCSRGraph& self) {
            float* data_ptr = const_cast<float*>(self.get_edge_weights());
            // Length zero rather than a throw, so a caller can branch on
            // .size instead of catching.
            uint32_t size = data_ptr ? self.get_edge_capacity() : 0;

            return as_readonly(py::array_t<float>(
                {size},
                {sizeof(float)},
                data_ptr,
                py::cast(self)
            ));
        }, "Zero-copy view of the per-slot edge weight, parallel to get_edges(). "
           "Empty unless the graph was built with store_weights=True.")

        .def_property_readonly("has_weights", &PCSRGraph::has_weights,
            "True if this graph carries per-edge weights.")

        .def("get_vertex_starts", [](PCSRGraph& self) {
            uint32_t* data_ptr = const_cast<uint32_t*>(self.get_vertex_starts());
            uint32_t size = self.get_num_vertices();

            return as_readonly(py::array_t<uint32_t>(
                {size},
                {sizeof(uint32_t)},
                data_ptr,
                py::cast(self)
            ));
        }, "Zero-copy view of the expired prefix length per region (length V). "
           "All zero until expire_before() is called; a vertex's live run starts "
           "at get_vertex_offsets()[v] + get_vertex_starts()[v].")

        .def("get_vertex_counts", [](PCSRGraph& self) {
            uint32_t* data_ptr = const_cast<uint32_t*>(self.get_vertex_counts());
            uint32_t size = self.get_num_vertices();

            return as_readonly(py::array_t<uint32_t>(
                {size},
                {sizeof(uint32_t)},
                data_ptr,
                py::cast(self)
            ));
        }, "Zero-copy view of the live out-degree of each vertex (length V).")

        .def("get_edges", [](PCSRGraph& self) {
            void* data_ptr = const_cast<void*>(static_cast<const void*>(self.get_edges()));
            uint32_t size = self.get_edge_capacity();

            return as_readonly(py::array_t<uint32_t>(
                {size, (uint32_t)2},
                {sizeof(uint32_t) * 2, sizeof(uint32_t)},
                static_cast<uint32_t*>(data_ptr),
                py::cast(self)
            ));
        }, "Zero-copy (capacity, 2) view of the whole PMA: column 0 is the target "
           "vertex (EMPTY_GAP = 4294967295 marks a gap), column 1 the timestamp.")

        .def("get_neighbors", [](PCSRGraph& self, uint32_t v) {
            if (v >= self.get_num_vertices()) {
                throw std::out_of_range("vertex id out of range");
            }
            const TemporalEdge* base = self.get_edges() + self.get_vertex_offsets()[v]
                                     + self.get_vertex_starts()[v];
            uint32_t degree = self.get_degree(v);

            return as_readonly(py::array_t<uint32_t>(
                {degree, (uint32_t)2},
                {sizeof(uint32_t) * 2, sizeof(uint32_t)},
                reinterpret_cast<const uint32_t*>(base),
                py::cast(self)
            ));
        }, py::arg("vertex"),
           "Zero-copy (degree, 2) view of one vertex's adjacency run. Gap-free: "
           "regions are left-packed, so this is exactly the live edges.")

        .def("to_coo", [](PCSRGraph& self) {
            // A real copy, unlike the views above: PyTorch Geometric wants
            // separate contiguous src/dst/time vectors, which is a different
            // layout from the PMA. O(E) and explicit rather than hidden.
            const size_t total = static_cast<size_t>(self.get_num_edges());
            py::array_t<uint32_t> src(total), dst(total), ts(total);

            auto* sp = src.mutable_data();
            auto* dp = dst.mutable_data();
            auto* tp = ts.mutable_data();

            const uint32_t* offsets = self.get_vertex_offsets();
            const uint32_t* counts = self.get_vertex_counts();
            const uint32_t* starts = self.get_vertex_starts();
            const TemporalEdge* edges = self.get_edges();

            py::array_t<EdgeRelation> rel(total);
            auto* rp = rel.mutable_data();
            const EdgeRelation* relations = self.get_edge_relations();

            const float* weights = self.get_edge_weights();
            py::array_t<float> weight(weights ? total : 0);
            auto* wp = weight.mutable_data();

            size_t k = 0;
            for (uint32_t v = 0; v < self.get_num_vertices(); ++v) {
                const uint32_t live = offsets[v] + starts[v];
                for (uint32_t i = 0; i < counts[v]; ++i) {
                    const TemporalEdge& e = edges[live + i];
                    sp[k] = v;
                    dp[k] = e.target_node;
                    tp[k] = e.timestamp;
                    rp[k] = relations[live + i];
                    if (weights) wp[k] = weights[live + i];
                    ++k;
                }
            }
            return py::make_tuple(src, dst, ts, rel, weight);
        }, "Materialise the graph as (src, dst, timestamp, relation, weight) arrays for "
           "PyTorch Geometric. This copies; the get_* views do not.")

        .def("get_degree", &PCSRGraph::get_degree, py::arg("vertex"))

        .def_property_readonly("num_vertices", &PCSRGraph::get_num_vertices)
        .def_property_readonly("edge_capacity", &PCSRGraph::get_edge_capacity)
        .def_property_readonly("num_edges", &PCSRGraph::get_num_edges)
        .def_property_readonly("arena_used", &PCSRGraph::get_arena_used)
        .def_property_readonly("rebalance_count", &PCSRGraph::get_rebalance_count)
        .def_property_readonly("resize_count", &PCSRGraph::get_resize_count)
        .def_property_readonly("slots_rewritten", &PCSRGraph::get_slots_rewritten)
        .def_property_readonly("expired_edges", &PCSRGraph::get_expired_edges,
            "Lifetime edges removed by expire_before().")
        .def_property_readonly("dead_slots", &PCSRGraph::get_dead_slots,
            "Expired slots still held, awaiting the next rebalance of their window.")

        .def("__repr__", [](const PCSRGraph& self) {
            return "<PCSRGraph vertices=" + std::to_string(self.get_num_vertices()) +
                   " edges=" + std::to_string(self.get_num_edges()) +
                   " capacity=" + std::to_string(self.get_edge_capacity()) + ">";
        });

    // ---- streaming ingestion over the lock-free queue ----------------------
    py::class_<StreamingIngestor>(m, "StreamingIngestor")
        .def(py::init([](PCSRGraph& graph, size_t queue_capacity) {
            return new StreamingIngestor(graph, queue_capacity);
        }), py::arg("graph"), py::arg("queue_capacity") = 65536,
            py::keep_alive<1, 2>(),  // the ingestor holds a reference to the graph
            "Streams events into a graph across a lock-free SPSC queue, with a "
            "dedicated C++ consumer thread. Lets a Python producer decode the "
            "next chunk while the previous one is still being inserted.")

        .def("start", [](StreamingIngestor& self) {
            py::gil_scoped_release release;
            self.start();
        }, "Spawn the consumer thread.")

        .def("push_batch", [](StreamingIngestor& self,
                              py::array_t<uint32_t, py::array::c_style | py::array::forcecast> src,
                              py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dst,
                              py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ts,
                              py::object relations,
                              py::object weights) {
            if (src.size() != dst.size() || src.size() != ts.size()) {
                throw std::invalid_argument("src, dst and timestamp must be the same length");
            }
            const uint32_t* sp = src.data();
            const uint32_t* dp = dst.data();
            const uint32_t* tp = ts.data();
            const size_t n = static_cast<size_t>(src.size());

            const EdgeRelation* rp = nullptr;
            py::array_t<EdgeRelation, py::array::c_style | py::array::forcecast> rel_array;
            if (!relations.is_none()) {
                rel_array = relations.cast<py::array_t<EdgeRelation,
                                py::array::c_style | py::array::forcecast>>();
                if (static_cast<size_t>(rel_array.size()) != n) {
                    throw std::invalid_argument("relation array length must match");
                }
                rp = rel_array.data();
            }

            const float* wp = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> weight_array;
            if (!weights.is_none()) {
                weight_array = weights.cast<py::array_t<float,
                                   py::array::c_style | py::array::forcecast>>();
                if (static_cast<size_t>(weight_array.size()) != n) {
                    throw std::invalid_argument("weight array length must match");
                }
                wp = weight_array.data();
            }

            // GIL released for the whole push: the producer blocks on
            // back-pressure, and holding the GIL there would stall every other
            // Python thread including the one decoding the next chunk.
            py::gil_scoped_release release;
            for (size_t i = 0; i < n; ++i) {
                self.push({sp[i], dp[i], tp[i], rp ? rp[i] : RELATION_UNKNOWN,
                           wp ? wp[i] : 0.0f});
            }
            return n;
        }, py::arg("src"), py::arg("dst"), py::arg("timestamp"),
           py::arg("relation") = py::none(), py::arg("weight") = py::none())

        .def("drain", [](StreamingIngestor& self) {
            py::gil_scoped_release release;
            self.drain();
        }, "Block until the consumer has caught up.")

        .def("stop", [](StreamingIngestor& self) {
            py::gil_scoped_release release;
            self.stop();
        }, "Drain the queue and join the consumer thread.")

        .def("__enter__", [](StreamingIngestor& self) {
            { py::gil_scoped_release release; self.start(); }
            return &self;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](StreamingIngestor& self, py::object, py::object, py::object) {
            py::gil_scoped_release release;
            self.stop();
            return false;
        })

        .def_property_readonly("running", &StreamingIngestor::is_running)
        .def_property_readonly("pushed", &StreamingIngestor::get_pushed)
        .def_property_readonly("consumed", &StreamingIngestor::get_consumed)
        .def_property_readonly("producer_spins", &StreamingIngestor::get_producer_spins)
        .def_property_readonly("consumer_idle_polls", &StreamingIngestor::get_consumer_idle_polls)
        .def_property_readonly("queue_capacity", &StreamingIngestor::get_queue_capacity);

    m.attr("EMPTY_GAP") = EMPTY_GAP;
    m.attr("RELATION_UNKNOWN") = RELATION_UNKNOWN;
}
