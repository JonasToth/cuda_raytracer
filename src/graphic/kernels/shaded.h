#ifndef SHADED_H_KK3H1DCZ
#define SHADED_H_KK3H1DCZ


#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "graphic/triangle.h"
#include <gsl/gsl>


/// Calculate the whole shading formular for one channel
/// This is C-Style, since it must run on the gpu as well, therefor no nice vectors
// CUCALL color phong_shading(const phong_material* m,
// const light_source* lights, std::size_t light_count,
// const coord& ray_direction, const intersect& hit);

CUCALL inline float clamp(float lowest, float value, float highest)
{
    if (value < lowest)
        return lowest;
    else if (value > highest)
        return highest;
    else
        return value;
}


#ifdef __CUDACC__
#include "management/surface_raii.h"

template <typename ShadingStyleTag>
__global__ void trace_triangles_shaded(cudaSurfaceObject_t surface, camera c,
                                       gsl::span<const triangle> triangles,
                                       gsl::span<const light_source> lights,
                                       ShadingStyleTag sst);
#include "graphic/kernels/shaded.inl"

#else

#include "management/memory_surface.h"
template <typename ShadingStyleTag>
void trace_triangles_shaded(memory_surface& surface, camera c,
                            gsl::span<const triangle> triangles,
                            gsl::span<const light_source> lights, ShadingStyleTag sst);
#endif

#endif /* end of include guard: SHADED_H_KK3H1DCZ */
