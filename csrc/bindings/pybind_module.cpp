#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/pcsr_graph.hpp"

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
        .def(py::init<uint32_t, uint32_t, size_t>(),
             py::arg("max_vertices"),
             py::arg("initial_edge_capacity"),
             py::arg("arena_bytes") = 128 * 1024 * 1024)

        .def("insert_edge", &PCSRGraph::insert_edge,
             py::arg("src"), py::arg("dst"), py::arg("timestamp"),
             py::arg("relation") = RELATION_UNKNOWN)

        .def("insert_edges", [](PCSRGraph& self,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> src,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dst,
                                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ts,
                                py::object relations) {
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

            py::gil_scoped_release release;
            for (size_t i = 0; i < n; ++i) {
                self.insert_edge(sp[i], dp[i], tp[i],
                                 rp ? rp[i] : RELATION_UNKNOWN);
            }
            return n;
        }, py::arg("src"), py::arg("dst"), py::arg("timestamp"),
           py::arg("relation") = py::none(),
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
            const TemporalEdge* base = self.get_edges() + self.get_vertex_offsets()[v];
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
            const TemporalEdge* edges = self.get_edges();

            py::array_t<EdgeRelation> rel(total);
            auto* rp = rel.mutable_data();
            const EdgeRelation* relations = self.get_edge_relations();

            size_t k = 0;
            for (uint32_t v = 0; v < self.get_num_vertices(); ++v) {
                for (uint32_t i = 0; i < counts[v]; ++i) {
                    const TemporalEdge& e = edges[offsets[v] + i];
                    sp[k] = v;
                    dp[k] = e.target_node;
                    tp[k] = e.timestamp;
                    rp[k] = relations[offsets[v] + i];
                    ++k;
                }
            }
            return py::make_tuple(src, dst, ts, rel);
        }, "Materialise the graph as (src, dst, timestamp, relation) arrays for PyTorch "
           "Geometric. This copies; the get_* views do not.")

        .def("get_degree", &PCSRGraph::get_degree, py::arg("vertex"))

        .def_property_readonly("num_vertices", &PCSRGraph::get_num_vertices)
        .def_property_readonly("edge_capacity", &PCSRGraph::get_edge_capacity)
        .def_property_readonly("num_edges", &PCSRGraph::get_num_edges)
        .def_property_readonly("arena_used", &PCSRGraph::get_arena_used)
        .def_property_readonly("rebalance_count", &PCSRGraph::get_rebalance_count)
        .def_property_readonly("resize_count", &PCSRGraph::get_resize_count)
        .def_property_readonly("slots_rewritten", &PCSRGraph::get_slots_rewritten)

        .def("__repr__", [](const PCSRGraph& self) {
            return "<PCSRGraph vertices=" + std::to_string(self.get_num_vertices()) +
                   " edges=" + std::to_string(self.get_num_edges()) +
                   " capacity=" + std::to_string(self.get_edge_capacity()) + ">";
        });

    m.attr("EMPTY_GAP") = EMPTY_GAP;
    m.attr("RELATION_UNKNOWN") = RELATION_UNKNOWN;
}
