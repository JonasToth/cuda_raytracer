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


#ifdef __CUDACC__ // GPU Raytracing
#include "management/surface_raii.h"

template <typename Camera, typename ShadingStyleTag, typename ShadowTag>
__global__ void trace_triangles_shaded(cudaSurfaceObject_t surface, Camera c,
                                       gsl::span<const triangle> triangles,
                                       gsl::span<const light_source> lights,
                                       ShadingStyleTag sst, ShadowTag st);
#include "graphic/kernels/shaded.inl"

#else // CPU raytracing
#include "management/memory_surface.h"

template <typename Camera, typename ShadingStyleTag, typename ShadowTag>
void trace_triangles_shaded(memory_surface& surface, Camera c,
                            gsl::span<const triangle> triangles,
                            gsl::span<const light_source> lights, ShadingStyleTag sst,
                            ShadowTag st);

#include "graphic/kernels/shaded.cpp.inl"
#endif

#endif /* end of include guard: SHADED_H_KK3H1DCZ */
