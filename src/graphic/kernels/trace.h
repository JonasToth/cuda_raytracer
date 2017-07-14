#ifndef TRACE_H_ORLZJOGP
#define TRACE_H_ORLZJOGP

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "graphic/triangle.h"
#include "management/surface_raii.h"

__global__ void trace_single_triangle(cudaSurfaceObject_t surface, const triangle* T,
                                      std::size_t width, std::size_t height);

__global__ void trace_many_triangles_with_camera(cudaSurfaceObject_t surface, camera c,
                                                 const triangle* triangles,
                                                 int n_triangles, int width, int height);

#include "trace.inl"

#endif /* end of include guard: TRACE_H_ORLZJOGP */
