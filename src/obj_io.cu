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
    Expects(file_name != nullptr);
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

struct VertexData 
{
    // pointers to the vertices
    const coord* P0;
    const coord* P1;
    const coord* P2;

    // indices to the vertices
    int i0;
    int i1;
    int i2;
};

VertexData vertex_information(const thrust::device_vector<coord>& vertices,
                              const std::vector<tinyobj::index_t>& indices,
                              const std::size_t index_offset)
{
    // all indices of the face
    const auto idx0 = indices[index_offset + 0].vertex_index;
    const auto idx1 = indices[index_offset + 1].vertex_index;
    const auto idx2 = indices[index_offset + 2].vertex_index;
    
    Expects(idx0 < vertices.size() && idx0 >= 0);
    Expects(idx1 < vertices.size() && idx1 >= 0);
    Expects(idx2 < vertices.size() && idx2 >= 0);

    VertexData p;
    p.P0 = (&vertices[idx0]).get();
    p.P1 = (&vertices[idx1]).get();
    p.P2 = (&vertices[idx2]).get();
    p.i0 = idx0;
    p.i1 = idx1;
    p.i2 = idx2;
    
    return p;
}

struct NormalData 
{
    // vertex normals
    const coord* n0;
    const coord* n1;
    const coord* n2;
    const coord* fn; // face normal
    
    // indices for normals
    int i0;
    int i1;
    int i2;
    int f;
};

// will insert face normal if necessary, otherwise just get all pointers right
NormalData normal_information(const thrust::device_vector<coord>& vertices,
                              const std::vector<tinyobj::index_t>& indices,
                              const VertexData vd,
                              const std::size_t index_offset,
                              thrust::device_vector<coord>& normals)
{
    NormalData nd;

#if 0
    std::clog << "Input: " << vd.i0 << " " << vd.i1 << " " << vd.i2 << std::endl;

    for(const auto& i: indices)
        std::clog << i.normal_index << std::endl;
#endif

    // all vertex normals
    const auto n_idx0 = indices[index_offset + 0].normal_index;
    const auto n_idx1 = indices[index_offset + 1].normal_index;
    const auto n_idx2 = indices[index_offset + 2].normal_index;

    // if normals not in file
        // calculate face normal
        // insert face normal
        // set all normals and indices to the inserted normal
        // CHECK
    // else // normals in file
        // if all normals identical
            // set all pointers and indices to first normal
        // else  // normals are not identical
            // set vertex normals to the one in file
            // calculate face normal from vertices

    // AXIOM: either all normals are set, or none are!
    if(n_idx0 == -1) // vertex normals not in file
    {
        Expects(n_idx1 == -1);
        Expects(n_idx2 == -1);

        // calculate face normal
        const coord p0 = vertices[vd.i0];
        const coord p1 = vertices[vd.i1];
        const coord p2 = vertices[vd.i2];
        const coord n  = normalize(cross(p1 - p0, p2 - p1));
        // push back normal, and get the last index (the created normal)
        normals.push_back(n);
        const auto fn_idx = normals.size() - 1;
        const auto* normal_ptr  = (&normals[fn_idx]).get();

        nd.n0 = normal_ptr;
        nd.n1 = normal_ptr;
        nd.n2 = normal_ptr;
        nd.fn = normal_ptr;

        nd.i0 = fn_idx;
        nd.i1 = fn_idx;
        nd.i2 = fn_idx;
        nd.f  = fn_idx;
    }
    else  // vertex normals in file
    {
        if((n_idx0 == n_idx1) && (n_idx1 == n_idx2)) // normals are identical
        {
            // all normals (including face normal) are the same
            const auto* normal_ptr = (&normals[n_idx0]).get();
            nd.n0 = normal_ptr;
            nd.n1 = normal_ptr;
            nd.n2 = normal_ptr;
            nd.fn = normal_ptr;

            nd.i0 = n_idx0;
            nd.i1 = n_idx0;
            nd.i2 = n_idx0;
            nd.f  = n_idx0;
        }
        else // normals are different
        {
            // vertex normals from file
            nd.n0 = (&normals[n_idx0]).get();
            nd.n1 = (&normals[n_idx1]).get();
            nd.n2 = (&normals[n_idx2]).get();
            nd.i0 = n_idx0;
            nd.i1 = n_idx1;
            nd.i2 = n_idx2;

            // calculate face normal
            const coord p0 = vertices[n_idx0];
            const coord p1 = vertices[n_idx1];
            const coord p2 = vertices[n_idx2];
            const coord n  = normalize(cross(p1 - p0, p2 - p1));
            // push back normal, and get the last index (the created normal)
            normals.push_back(n);
            const auto fn_idx = normals.size() - 1;
            const auto* normal_ptr  = (&normals[fn_idx]).get();

            nd.fn = normal_ptr;
            nd.f  = fn_idx;
        }

    }
    
    Ensures(nd.i0 < normals.size() && nd.i0 >= 0);
    Ensures(nd.i1 < normals.size() && nd.i1 >= 0);
    Ensures(nd.i2 < normals.size() && nd.i2 >= 0);
    Ensures(nd.n0 != nullptr);
    Ensures(nd.n1 != nullptr);
    Ensures(nd.n2 != nullptr);
    Ensures(nd.fn != nullptr);

    return nd;
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
            const auto td = vertex_information(vertices, shape.mesh.indices, 
                                               index_offset);
            // updates normals if necessary
            const auto nd = normal_information(vertices, shape.mesh.indices,
                                               td, index_offset, normals);

            triangle t(td.P0, td.P1, td.P2, nd.fn);
            t.p0_normal(nd.n0);
            t.p1_normal(nd.n1);
            t.p2_normal(nd.n2);

            // WARN: this writes the pointer on the device, as material pointer.
            // if you derefence material on the cpu, that results in a segfault
            // on cuda devices!
            const phong_material* m_ptr = shape.mesh.material_ids[f] < 0 
                                          ? nullptr
                                          : (&materials[shape.mesh.material_ids[f]]).get();
            t.material(m_ptr);

            // add triangle to world
            triangles.push_back(t);
                
            index_offset+= shape.mesh.num_face_vertices[f];
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
    Expects(__normals.size() > 0);
}


