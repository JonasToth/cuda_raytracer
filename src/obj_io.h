#ifndef OBJ_IO_H_ZHYWUHRN
#define OBJ_IO_H_ZHYWUHRN

#include <string>
#include <thrust/device_vector.h>

#include "tinyobjloader/tiny_obj_loader.h"
#include "triangle.h"

/// Holds the World Geometry in a device vector and handles loading it from file.
///
/// It uses tiny_obj_loader -> https://github.com/syoyo/tinyobjloader
/// to read in .obj-files and uses it conventions with minor modifications.
///
/// Textures and Normales currently not supported
class WorldGeometry {
public:
    /// Will leave everything empty, manually load with `load`
    WorldGeometry() = default;
    /// Will already load data from disk with `load`
    //WorldGeometry(const std::string& file_name) { load(file_name); }

    // Copy and move operations currently forbidden, might be allowed later.
    WorldGeometry(const WorldGeometry&) = delete;
    WorldGeometry& operator=(const WorldGeometry&) = delete;

    WorldGeometry(WorldGeometry&&) = delete;
    WorldGeometry& operator=(WorldGeometry&&) = delete;


    /// Load data from disk and upload to thrust::device, throws exception on data error.
    void load(const std::string& file_name);

private:
    thrust::device_vector<coord> __vertices;        ///< all vertices in the world
    thrust::device_vector<triangle> __triangles;    ///< references the __vertices
};


#endif /* end of include guard: OBJ_IO_H_ZHYWUHRN */
