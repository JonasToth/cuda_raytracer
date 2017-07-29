#ifndef WORLD_DEPTH_H_OJHV0R12
#define WORLD_DEPTH_H_OJHV0R12

#include "graphic/kernels/trace.h"
#include "graphic/kernels/utility.h"

#ifdef __CUDACC__ // GPU Raytracing
template <typename Camera>
inline void raytrace_many_cuda(cudaSurfaceObject_t Surface, Camera c,
                               gsl::span<const triangle> triangles)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(Surface, c.width(), c.height());
    trace_many_triangles_with_camera<<<dimGrid, dimBlock>>>(
        Surface, c, triangles.data(), triangles.size(), c.width(), c.height());
    cudaDeviceSynchronize();
}


inline void raytrace_cuda(cudaSurfaceObject_t Surface, int width, int height,
                          const triangle* T)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x) / dimBlock.x, (height + dimBlock.y) / dimBlock.y);
    trace_single_triangle<<<dimGrid, dimBlock>>>(Surface, T, width, height);
    cudaDeviceSynchronize();
}


#else // CPU Raytracing

template <typename Camera>
inline void raytrace_many_cuda(memory_surface& s, Camera c,
                               gsl::span<const triangle> triangles)
{
    trace_many_triangles_with_camera(s, c, triangles.data(), triangles.size());
}

inline void raytrace_cuda(memory_surface& s, const triangle& t)
{
    trace_single_triangle(s, t);
}

#endif

#endif /* end of include guard: WORLD_DEPTH_H_OJHV0R12 */
