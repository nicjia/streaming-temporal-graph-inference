#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/pcsr_graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(graph_engine, m) {
    m.doc() = "Zero-allocation High-Throughput Graph Engine";

    py::class_<PCSRGraph>(m, "PCSRGraph")
        .def(py::init<uint32_t, uint32_t, size_t>(), 
             py::arg("max_vertices"), 
             py::arg("initial_edge_capacity"),
             py::arg("arena_bytes") = 128 * 1024 * 1024)
             
        .def("insert_edge", &PCSRGraph::insert_edge,
             py::arg("src"), py::arg("dst"), py::arg("timestamp"))
        
        .def("get_vertex_offsets", [](PCSRGraph& self) {
            uint32_t* data_ptr = const_cast<uint32_t*>(self.get_vertex_offsets());
            uint32_t size = self.get_num_vertices() + 1;
            
            return py::array_t<uint32_t>(
                {size}, 
                {sizeof(uint32_t)}, 
                data_ptr, 
                py::cast(self) 
            );
        })

        .def("get_edges", [](PCSRGraph& self) {
            void* data_ptr = const_cast<void*>(static_cast<const void*>(self.get_edges()));
            uint32_t size = self.get_edge_capacity();
        
            return py::array_t<uint32_t>(
                {size, (uint32_t)2}, 
                {sizeof(uint32_t) * 2, sizeof(uint32_t)}, 
                static_cast<uint32_t*>(data_ptr), 
                py::cast(self) 
            );
        });
}