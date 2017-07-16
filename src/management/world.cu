#ifdef __CUDACC__
#include "management/world.h"
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <algorithm>
#include <gsl/gsl>
#include <iostream>
#include <stdexcept>

#define DEBUG_OUTPUT 1

world_geometry::world_geometry() = default;
world_geometry::world_geometry(const std::string& file_name) { load(file_name); }


// Wrap the tiny_obj library and do the internal data structuring
namespace __detail
{
struct loaded_data {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};


loaded_data load(const char* file_name)
{
    Expects(file_name != nullptr);
    loaded_data result;
    std::string err;
    bool ret = tinyobj::LoadObj(&result.attrib, &result.shapes, &result.materials, &err,
                                file_name);

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (ret == false) {
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
    Expects(vertices.size() % 3 == 0ul);

    thrust::host_vector<coord> v;
    v.reserve(vertices.size() / 3);

    /// See data format for tinyobjloader
    for (std::size_t i = 0; i < vertices.size(); i += 3)
        v.push_back({vertices[i], vertices[i + 1], vertices[i + 2]});

    return v;
}

thrust::host_vector<phong_material>
build_materials(const std::vector<tinyobj::material_t>& materials)
{
    thrust::host_vector<phong_material> m;
    m.reserve(materials.size());

    for (const auto& mat : materials) {
        m.push_back(phong_material(static_cast<const float*>(mat.specular),
                                   static_cast<const float*>(mat.diffuse),
                                   static_cast<const float*>(mat.ambient),
                                   static_cast<float>(mat.shininess)));
    }

    return m;
}

struct VertexData {
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

    Expects(static_cast<std::size_t>(idx0) < vertices.size() && idx0 >= 0);
    Expects(static_cast<std::size_t>(idx1) < vertices.size() && idx1 >= 0);
    Expects(static_cast<std::size_t>(idx2) < vertices.size() && idx2 >= 0);

    VertexData p;
    p.i0 = idx0;
    p.i1 = idx1;
    p.i2 = idx2;

    return p;
}

struct NormalData {
    // indices for normals
    int i0;
    int i1;
    int i2;
    int f;
};

// will insert face normal if necessary, otherwise just get all pointers right
NormalData normal_information(const thrust::device_vector<coord>& vertices,
                              const std::vector<tinyobj::index_t>& indices,
                              const VertexData vd, const std::size_t index_offset,
                              thrust::device_vector<coord>& normals)
{
    NormalData nd;

#ifdef NNDEBUG
    std::clog << "Input: " << vd.i0 << " " << vd.i1 << " " << vd.i2 << std::endl;

    for (const auto& i : indices)
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
    if (n_idx0 == -1) // vertex normals not in file
    {
        Expects(n_idx1 == -1);
        Expects(n_idx2 == -1);

        // calculate face normal
        const coord p0 = vertices[vd.i0];
        const coord p1 = vertices[vd.i1];
        const coord p2 = vertices[vd.i2];
        const coord n = normalize(cross(p1 - p0, p2 - p1));
        // push back normal, and get the last index (the created normal)
        normals.push_back(n);
        const auto fn_idx = normals.size() - 1;

        nd.i0 = fn_idx;
        nd.i1 = fn_idx;
        nd.i2 = fn_idx;
        nd.f = fn_idx;
    } else // vertex normals in file
    {
        if ((n_idx0 == n_idx1) && (n_idx1 == n_idx2)) // normals are identical
        {
            // all normals (including face normal) are the same
            nd.i0 = n_idx0;
            nd.i1 = n_idx0;
            nd.i2 = n_idx0;
            nd.f = n_idx0;
        } else // normals are different
        {
            // vertex normals from file
            nd.i0 = n_idx0;
            nd.i1 = n_idx1;
            nd.i2 = n_idx2;

#if DEBUG_OUTPUT == 1
            std::clog << "ni0: " << nd.i0 << '\t' << "ni1: " << nd.i1 << '\t'
                      << "ni2: " << nd.i2 << '\t' << "nmax: " << normals.size() << '\t'
                      << '\t' << "vi0: " << vd.i0 << '\t' << "vi1: " << vd.i1 << '\t'
                      << "vi2: " << vd.i2 << '\t' << "vmax: " << vertices.size() << '\n';
#endif

            // calculate face normal
            const coord p0 = vertices[vd.i0];
            const coord p1 = vertices[vd.i1];
            const coord p2 = vertices[vd.i2];
            const coord n = normalize(cross(p1 - p0, p2 - p1));
            // push back normal, and get the last index (the created normal)
            normals.push_back(n);
            const auto fn_idx = normals.size() - 1;

            nd.f = fn_idx;
        }
    }

    Ensures(static_cast<std::size_t>(nd.i0) < normals.size() && nd.i0 >= 0);
    Ensures(static_cast<std::size_t>(nd.i1) < normals.size() && nd.i1 >= 0);
    Ensures(static_cast<std::size_t>(nd.i2) < normals.size() && nd.i2 >= 0);

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
    // temporarily store normal indices, because the normal vector could grow, and
    // invalidate all pointers
    thrust::host_vector<NormalData> face_normal_information;
    thrust::host_vector<VertexData> face_vertex_information;
    thrust::host_vector<const phong_material*> face_materials;

    for (const auto& shape : shapes) {
        // we will use only triangles
        Expects(std::all_of(std::begin(shape.mesh.num_face_vertices),
                            std::end(shape.mesh.num_face_vertices),
                            [](int i) { return i == 3; }));

        std::size_t index_offset = 0;
        // all faces of the shape
        for (std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const auto td = vertex_information(vertices, shape.mesh.indices, index_offset);
            // updates normals if necessary
            const auto nd = normal_information(vertices, shape.mesh.indices, td,
                                               index_offset, normals);

            // WARN: this writes the pointer on the device, as material pointer.
            // if you derefence material on the cpu, that results in a segfault
            // on cuda devices!
            const phong_material* m_ptr =
                shape.mesh.material_ids[f] < 0 ?
                    nullptr :
                    (&materials[shape.mesh.material_ids[f]]).get();

            // add triangle to world, and store the normal information for later
            // connection
            face_vertex_information.push_back(td);
            face_normal_information.push_back(nd);
            face_materials.push_back(m_ptr);

            index_offset += shape.mesh.num_face_vertices[f];
        }
    }

    Expects(face_normal_information.size() == face_vertex_information.size());
    Expects(face_normal_information.size() == face_materials.size());
    Expects(face_normal_information.size() > 0);

    // connect all normals with the triangle, the normal vector is expected to not grow
    // anymore
    for (std::size_t i = 0; i < face_normal_information.size(); ++i) {
        const auto& nd = face_normal_information[i];
        const auto& td = face_vertex_information[i];

        triangle t((&vertices[td.i0]).get(), (&vertices[td.i1]).get(),
                   (&vertices[td.i2]).get(), (&normals[nd.f]).get());
        t.p0_normal((&normals[nd.i0]).get());
        t.p1_normal((&normals[nd.i1]).get());
        t.p2_normal((&normals[nd.i2]).get());
        t.material(face_materials[i]);

        triangles.push_back(t);
    }

    Ensures(triangles.size() > 0);
    Ensures(normals.size() > 0);

    return triangles;
}
} // namespace __detail



void world_geometry::load(const std::string& file_name)
{
    const auto data = __detail::load(file_name.c_str());
    __shape_count = data.shapes.size();

    // Handle all Vertices
    __vertices = __detail::build_coords(data.attrib.vertices);
    Expects(__vertices.size() == data.attrib.vertices.size() / 3ul);

    // Handle all normals
    __normals = __detail::build_coords(data.attrib.normals);

    // Handle all Materials
    __materials = __detail::build_materials(data.materials);
    Expects(__materials.size() == data.materials.size());

    // Connect the triangles and give their surfaces a material, creates normals if
    // necessary!
    __triangles = __detail::build_faces(data.shapes, __vertices, __normals, __materials);
    Expects(__normals.size() > 0ul);
}


void world_geometry::add_light(phong_light l, coord position)
{
    __lights.push_back(light_source(l, position));
}


world_geometry::data_handle::data_handle(const thrust::device_vector<coord>& vert,
                                         const thrust::device_vector<coord>& norm,
                                         const thrust::device_vector<triangle>& tria,
                                         const thrust::device_vector<phong_material>& mat,
                                         const thrust::device_vector<light_source>& light,
                                         const camera c)
  : vertices{vert.data().get()}
  , vertex_count{vert.size()}
  , normals{norm.data().get()}
  , normal_count{norm.size()}
  , triangles{tria.data().get()}
  , triangle_count{tria.size()}
  , materials{mat.data().get()}
  , material_count{mat.size()}
  , lights{light.data().get()}
  , light_count{light.size()}
  , cam{c}
{
}

world_geometry::data_handle world_geometry::handle() const noexcept
{
    return data_handle(__vertices, __normals, __triangles, __materials, __lights, __c);
}
