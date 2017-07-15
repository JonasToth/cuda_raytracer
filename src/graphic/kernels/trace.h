#ifndef TRACE_H_ORLZJOGP
#define TRACE_H_ORLZJOGP

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "graphic/triangle.h"
#include "management/memory_surface.h"
#include "management/surface_raii.h"

__global__ void trace_single_triangle(cudaSurfaceObject_t surface, const triangle* T,
                                      std::size_t width, std::size_t height);

void trace_single_triangle(memory_surface& surface, const triangle& T);


__global__ void trace_many_triangles_with_camera(cudaSurfaceObject_t surface, camera c,
                                                 const triangle* triangles,
                                                 int n_triangles, int width, int height);
void trace_many_triangles_with_camera(memory_surface& surface, camera c,
                                      const triangle* triangles, int n_triangles);

#ifdef __CUDACC__
#include "graphic/kernels/trace.inl"
#endif

#endif /* end of include guard: TRACE_H_ORLZJOGP */
