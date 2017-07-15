#ifndef WORLD_DEPTH_H_OJHV0R12
#define WORLD_DEPTH_H_OJHV0R12

#include "graphic/kernels/trace.h"
#include "graphic/kernels/utility.h"

void raytrace_many_cuda(cudaSurfaceObject_t Surface, const camera& c,
                        const triangle* Triangles, int TriangleCount);
void raytrace_many_cuda(memory_surface& s, const camera& c, const triangle* triangles,
                        std::size_t triangle_count)
{
    trace_many_triangles_with_camera(s, c, triangles, triangle_count);
}

void raytrace_cuda(cudaSurfaceObject_t& Surface, int width, int height, const triangle* T);
void raytrace_cuda(memory_surface& s, const triangle& t) { trace_single_triangle(s, t); }

#ifdef __CUDACC__
#include "util/kernel_launcher/world_depth.inl"
#endif

#endif /* end of include guard: WORLD_DEPTH_H_OJHV0R12 */
