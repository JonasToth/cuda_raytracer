#include "obj_io.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <gsl/gsl>
#include <iostream>
#include <stdexcept>


WorldGeometry::WorldGeometry() = default;

// Wrap the tiny_obj library
namespace {
    struct loaded_data { 
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
    };

    loaded_data __load(const char* file_name) {
        loaded_data result;
        std::string err;
        bool ret = tinyobj::LoadObj(&result.attrib, &result.shapes, 
                                    &result.materials, &err, file_name);

        if(!err.empty()) {
            std::cerr << err << std::endl;
        }

        if(ret == false) {
            std::string error_message = "Could not read the file ";
            throw std::invalid_argument{error_message + file_name};
        }
        Ensures(result.attrib.vertices.size() % 3 == 0);
        return result;
    }
}

void WorldGeometry::load(const std::string& file_name) {
    // Load with the library
    const auto data = __load(file_name.c_str());
    const auto& v = data.attrib.vertices;
    const auto& s = data.shapes;

    Expects(v.size() % 3 == 0);

    // transform the data into local represenation
    __vertices.reserve(v.size() / 3); // per vertex 3 floats
    
    for(std::size_t i = 0; i < data.attrib.vertices.size(); i+= 3)
        __vertices.push_back({v[i], v[i+1], v[i+2]});

    // all shapes
    for(const auto& shape: s)
    {
        std::size_t index_offset = 0;
        // all faces of the shape
        for(std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            const auto fv = shape.mesh.num_face_vertices[f];
            Expects(fv == 3);

            // all indices of the face
            const auto idx0 = shape.mesh.indices[index_offset + 0].vertex_index;
            const auto idx1 = shape.mesh.indices[index_offset + 1].vertex_index;
            const auto idx2 = shape.mesh.indices[index_offset + 2].vertex_index;

            const thrust::device_ptr<coord> P0 = &__vertices[idx0];
            const thrust::device_ptr<coord> P1 = &__vertices[idx1];
            const thrust::device_ptr<coord> P2 = &__vertices[idx2];

            __triangles.push_back(triangle{P0.get(), 
                                           P1.get(), 
                                           P2.get()});
                
            index_offset+= fv;
        }
    }
}













