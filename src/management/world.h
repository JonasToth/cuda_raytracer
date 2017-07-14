#ifndef OBJ_IO_H_ZHYWUHRN
#define OBJ_IO_H_ZHYWUHRN


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <vector>

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/material.h"
#include "graphic/triangle.h"
#include "graphic/vector.h"

/// Holds the World Geometry in a device vector and handles loading it from file.
///
/// It uses tiny_obj_loader -> https://github.com/syoyo/tinyobjloader
/// to read in .obj-files and uses it conventions with minor modifications.
///
/// Textures and Normales currently not supported
/// Tight coupling between loading and geometry, might be removed later.
class world_geometry {
public:
    /// Will leave everything empty, manually load with `load`
    world_geometry();
    explicit world_geometry(const std::string& file_name);

    // Copy and move operations currently forbidden, might be allowed later.
    world_geometry(const world_geometry&) = delete;
    world_geometry& operator=(const world_geometry&) = delete;

    world_geometry(world_geometry&&) = delete;
    world_geometry& operator=(world_geometry&&) = delete;

    /// Load data from disk and upload to thrust::device, throws exception on data error.
    void load(const std::string& file_name);

    void add_light(phong_light l, coord position);

    void add_camera(camera c) noexcept { __c = c; }

    std::size_t vertex_count() const noexcept { return __vertices.size(); }
    std::size_t normal_count() const noexcept { return __normals.size(); }
    std::size_t material_count() const noexcept { return __materials.size(); }
    std::size_t triangle_count() const noexcept { return __triangles.size(); }
    std::size_t shape_count() const noexcept { return __shape_count; }
    std::size_t light_count() const noexcept { return __lights.size(); }

    const thrust::device_vector<coord>& vertices() const noexcept { return __vertices; }
    const thrust::device_vector<coord>& normals() const noexcept { return __normals; }
    const thrust::device_vector<phong_material>& materials() const noexcept { return __materials; }
    const thrust::device_vector<triangle>& triangles() const noexcept { return __triangles; }
    const thrust::device_vector<light_source>& lights() const noexcept { return __lights; }

    /// Helper struct to get the device pointers for all gpu data, 
    /// kernel call simplification
    struct data_handle {
        const coord* const vertices;      const std::size_t vertex_count;
        const coord* const normals;       const std::size_t normal_count;
        const triangle* const triangles;  const std::size_t triangle_count;

        const phong_material* const materials;    const std::size_t material_count;
        const light_source* const lights;         const std::size_t light_count;

        const camera cam;

        data_handle(const thrust::device_vector<coord>& vert,
                    const thrust::device_vector<coord>& norm,
                    const thrust::device_vector<triangle>& tria,
                    const thrust::device_vector<phong_material>& mat,
                    const thrust::device_vector<light_source>& light,
                    const camera c);
    };

    data_handle handle() const noexcept;

private:
    thrust::device_vector<coord> __vertices;            ///< vertices in the world
    thrust::device_vector<coord> __normals;             ///< normals for vertices and faces
    thrust::device_vector<phong_material> __materials;  ///< existing materials

    thrust::device_vector<triangle> __triangles;    ///< references the __vertices 
                                                    ///< and __materials
                                                    
    thrust::device_vector<light_source> __lights;   ///< lights in the scene
    camera __c;                                     ///< camera watching the scene

    std::size_t __shape_count;                      ///< number of shapes(objects) in scene
};


#endif /* end of include guard: OBJ_IO_H_ZHYWUHRN */
