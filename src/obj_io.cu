#include "obj_io.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <gsl/gsl>
#include <iostream>
#include <stdexcept>


world_geometry::world_geometry() = default;
world_geometry::world_geometry(const std::string& file_name) { load(file_name); }

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

void world_geometry::load(const std::string& file_name) {
    thrust::host_vector<coord> vertices;
    thrust::host_vector<triangle> triangles;
    thrust::host_vector<phong_material> materials;

    __detail::deserialize_geometry(file_name, vertices, triangles, materials, __shape_count);

    __vertices = vertices;
    __triangles = triangles;
    __materials = materials;
}

namespace __detail {
void deserialize_geometry(const std::string& file_name,
                         thrust::host_vector<coord>& vertices,
                         thrust::host_vector<triangle>& triangles,
                         thrust::host_vector<phong_material>& materials,
                         std::size_t& shape_count) {
    // Load with the library
    const auto data = __load(file_name.c_str());
    const auto& v = data.attrib.vertices;
    const auto& s = data.shapes;

    Expects(v.size() % 3 == 0);

    shape_count = data.shapes.size();

    // transform the data into local represenation
    vertices.reserve(v.size() / 3); // per vertex 3 floats
    
    for(std::size_t i = 0; i < data.attrib.vertices.size(); i+= 3)
        vertices.push_back({v[i], v[i+1], v[i+2]});

    // all shapes
    for(const auto& shape: s)
    {
        std::size_t index_offset = 0;
        // all faces of the shape
        for(std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            const auto fv = shape.mesh.num_face_vertices[f];
            if(fv != 3) 
                throw std::invalid_argument{"Found a polygon, need triangles only"};

            // all indices of the face
            const auto idx0 = shape.mesh.indices[index_offset + 0].vertex_index;
            const auto idx1 = shape.mesh.indices[index_offset + 1].vertex_index;
            const auto idx2 = shape.mesh.indices[index_offset + 2].vertex_index;

            const auto P0 = vertices[idx0];
            const auto P1 = vertices[idx1];
            const auto P2 = vertices[idx2];

            triangles.push_back(triangle{P0, P1, P2});
                
            index_offset+= fv;
        }
    }

    materials.reserve(data.materials.size());
    // all materials
    for(const auto& m: data.materials)
    {
        materials.push_back({m.specular[0], m.diffuse[0], m.ambient[0], m.shininess});
    }
}

}
