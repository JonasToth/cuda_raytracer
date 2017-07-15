#ifndef SHADED_H_KK3H1DCZ
#define SHADED_H_KK3H1DCZ


#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "graphic/triangle.h"
#include "management/memory_surface.h"
#include "management/surface_raii.h"


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


__global__ void trace_triangles_shaded(cudaSurfaceObject_t surface, camera c,
                                       const triangle* triangles, std::size_t n_triangles,
                                       const light_source* lights, std::size_t n_lights,
                                       int width, int height);

void trace_triangles_shaded(memory_surface& surface, camera c,
                            const triangle* triangles, std::size_t n_triangles,
                            const light_source* lights, std::size_t n_lights);
#ifdef __CUDACC__
#include "graphic/kernels/shaded.inl"
#endif

#endif /* end of include guard: SHADED_H_KK3H1DCZ */
