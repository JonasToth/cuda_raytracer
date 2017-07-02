#ifndef OBJ_IO_H_ZHYWUHRN
#define OBJ_IO_H_ZHYWUHRN


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <vector>
#include "vector.h"
#include "triangle.h"

/// Holds the World Geometry in a device vector and handles loading it from file.
///
/// It uses tiny_obj_loader -> https://github.com/syoyo/tinyobjloader
/// to read in .obj-files and uses it conventions with minor modifications.
///
/// Textures and Normales currently not supported
/// Tight coupling between loading and geometry, might be removed later.
class WorldGeometry {
public:
    /// Will leave everything empty, manually load with `load`
    WorldGeometry();

    // Copy and move operations currently forbidden, might be allowed later.
    WorldGeometry(const WorldGeometry&) = delete;
    WorldGeometry& operator=(const WorldGeometry&) = delete;

    WorldGeometry(WorldGeometry&&) = delete;
    WorldGeometry& operator=(WorldGeometry&&) = delete;


    /// Load data from disk and upload to thrust::device, throws exception on data error.
    void load(const std::string& file_name);

    std::size_t vertex_count() const noexcept { return __vertices.size(); }
    std::size_t triangle_count() const noexcept { return __triangles.size(); }

private:
    thrust::device_vector<coord> __vertices;        ///< all vertices in the world
    thrust::device_vector<triangle> __triangles;    ///< references the __vertices
    
};


#endif /* end of include guard: OBJ_IO_H_ZHYWUHRN */
