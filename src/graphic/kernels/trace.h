#ifndef TRACE_H_ORLZJOGP
#define TRACE_H_ORLZJOGP

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "graphic/triangle.h"

#ifdef __CUDACC__
#include "management/surface_raii.h"

__global__ void trace_single_triangle(cudaSurfaceObject_t surface, const triangle* T,
                                      std::size_t width, std::size_t height);
template <typename Camera>
__global__ void trace_many_triangles_with_camera(cudaSurfaceObject_t surface, camera c,
                                                 const triangle* triangles,
                                                 int n_triangles, int width, int height);
#include "graphic/kernels/trace.inl"

#else

#include "management/memory_surface.h"
void trace_single_triangle(memory_surface& surface, const triangle& t);
template <typename Camera>
void trace_many_triangles_with_camera(memory_surface& surface, camera c,
                                      const triangle* triangles, std::size_t n_triangles);

#endif

#endif /* end of include guard: TRACE_H_ORLZJOGP */
