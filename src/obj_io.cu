#include "obj_io.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <algorithm>
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
    Ensures(result.attrib.texcoords.size() % 2 == 0);
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
        // we will use only triangles
        Expects(std::all_of(std::begin(shape.mesh.num_face_vertices),
                            std::end(shape.mesh.num_face_vertices), 
                            [](int i) { return i == 3; }));

        std::size_t index_offset = 0;
        // all faces of the shape
        for(std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            const auto fv = shape.mesh.num_face_vertices[f];

            // all indices of the face
            const auto idx0 = shape.mesh.indices[index_offset + 0].vertex_index;
            const auto idx1 = shape.mesh.indices[index_offset + 1].vertex_index;
            const auto idx2 = shape.mesh.indices[index_offset + 2].vertex_index;
            
            Expects(idx0 < vertices.size() && idx0 >= 0);
            Expects(idx1 < vertices.size() && idx1 >= 0);
            Expects(idx2 < vertices.size() && idx2 >= 0);

            const auto* P0 = (&vertices[idx0]).get();
            const auto* P1 = (&vertices[idx1]).get();
            const auto* P2 = (&vertices[idx2]).get();

            // index of the normal
            auto nidx = shape.mesh.indices[index_offset].normal_index;
            bool calc_normal = false;
            // normals dont need to be saved -> calculate if not existing
            if(nidx < 0)
            {
                const coord p0 = vertices[idx0];
                const coord p1 = vertices[idx1];
                const coord p2 = vertices[idx2];
                normals.push_back(normalize(cross(p1 - p0, p2 - p1)));
                nidx = normals.size() - 1;
                calc_normal = true;
            }
            const auto* n  = (&normals[nidx]).get();

            triangle t(P0, P1, P2, n);

            if(!calc_normal) 
            {
                // all vertex normals
                const auto n_idx0 = shape.mesh.indices[index_offset + 0].normal_index;
                const auto n_idx1 = shape.mesh.indices[index_offset + 1].normal_index;
                const auto n_idx2 = shape.mesh.indices[index_offset + 2].normal_index;

                Expects(n_idx0 < normals.size() && n_idx0 >= 0);
                Expects(n_idx1 < normals.size() && n_idx1 >= 0);
                Expects(n_idx2 < normals.size() && n_idx2 >= 0);

                t.p0_normal((&normals[n_idx0]).get());
                t.p1_normal((&normals[n_idx1]).get());
                t.p2_normal((&normals[n_idx2]).get());
            }


            
            // WARN: this writes the pointer on the device, as material pointer.
            // if you derefence material on the cpu, that results in a segfault
            // on cuda devices!
            const phong_material* m_ptr = shape.mesh.material_ids[f] < 0 
                                          ? nullptr
                                          : (&materials[shape.mesh.material_ids[f]]).get();
            t.material(m_ptr);

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

    // Handle all Materials
    __materials = __detail::build_materials(data.materials);
    Expects(__materials.size() == data.materials.size());

    // Connect the triangles and give their surfaces a material, creates normals if
    // necessary!
    __triangles = __detail::build_faces(data.shapes, __vertices, __normals, __materials);
    //Expects(__normals.size() > 0);
}


