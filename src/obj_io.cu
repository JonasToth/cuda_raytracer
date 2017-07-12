#include "obj_io.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <gsl/gsl>
#include <iostream>
#include <stdexcept>


world_geometry::world_geometry() = default;
world_geometry::world_geometry(const std::string& file_name) { load(file_name); }


// Wrap the tiny_obj library and do the internal data structuring
namespace __detail {

struct loaded_data { 
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};


loaded_data load(const char* file_name) {
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
    Ensures(result.attrib.normals.size() % 3 == 0);
    Ensures(result.attrib.texcoords.size() % 3 == 0);
    return result;
}

thrust::host_vector<coord> build_coords(const std::vector<tinyobj::real_t>& vertices)
{
    Expects(vertices.size() % 3 == 0);

    thrust::host_vector<coord> v;
    v.reserve(vertices.size() / 3);

    /// See data format for tinyobjloader
    for(std::size_t i = 0; i < vertices.size(); i+= 3)
        v.push_back({vertices[i], vertices[i+1], vertices[i+2]});

    return v;
}

thrust::host_vector<phong_material> 
build_materials(const std::vector<tinyobj::material_t>& materials)
{
    thrust::host_vector<phong_material> m;
    m.reserve(materials.size());

    for(const auto& mat: materials)
    {
        m.push_back(phong_material(static_cast<const float*>(mat.specular), 
                                   static_cast<const float*>(mat.diffuse), 
                                   static_cast<const float*>(mat.ambient), 
                                   static_cast<float>(mat.shininess)));
    }
    
    return m;
}

/// Connects the vertices to triangles, assigns them a normal and material.
/// Normals will be calculated if necessary
thrust::device_vector<triangle> 
build_faces(const std::vector<tinyobj::shape_t>& shapes,
            const thrust::device_vector<coord>& vertices,
            thrust::device_vector<coord>& normals,
            const thrust::device_vector<phong_material>& materials)
{
    thrust::device_vector<triangle> triangles;
    for(const auto& shape: shapes)
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

            const auto* P0 = (&vertices[idx0]).get();
            const auto* P1 = (&vertices[idx1]).get();
            const auto* P2 = (&vertices[idx2]).get();
            
            triangle t(P0, P1, P2);
            
            // WARN: this writes the pointer on the device, as material pointer.
            // if you derefence material on the cpu, that results in a segfault
            // on cuda devices!
            auto m_ptr = &materials[shape.mesh.material_ids[f]];
            t.material(m_ptr.get());

            triangles.push_back(t);
                
            index_offset+= fv;
        }
    }
    return triangles;
}
} // namespace __detail



void world_geometry::load(const std::string& file_name) {
    const auto data = __detail::load(file_name.c_str());
    __shape_count = data.shapes.size();

    // Handle all Vertices
    __vertices = __detail::build_coords(data.attrib.vertices);
    Expects(__vertices.size() == data.attrib.vertices.size() / 3);

    // Handle all normals
    __normals = __detail::build_coords(data.attrib.normals);
    Expects(__normals.size() == data.attrib.normals.size() / 3);

    // Handle all Materials
    __materials = __detail::build_materials(data.materials);
    Expects(__materials.size() == data.materials.size());

    // Connect the triangles and give their surfaces a material, creates normals if
    // necessary!
    __triangles = __detail::build_faces(data.shapes, __vertices, __normals, __materials);
}


