#ifndef WORLD_SHADING_H_E0JFD9ZA
#define WORLD_SHADING_H_E0JFD9ZA


#include "graphic/kernels/shaded.h"
#include "graphic/kernels/utility.h"
#include "management/world.h"

#ifdef __CUDACC__ // GPU Raytracing
template <typename ShadowTag, typename Camera>
void render_flat(cudaSurfaceObject_t surface, Camera c, world_geometry::data_handle dh)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, c.width(), c.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(
        surface, c, dh.triangles.data(), dh.triangles.size(), dh.lights.data(),
        dh.lights.size(), flat_shading_tag{}, ShadowTag{});
    cudaDeviceSynchronize();
}
template <typename ShadowTag, typename Camera>
void render_smooth(cudaSurfaceObject_t surface, Camera c, world_geometry::data_handle dh)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, c.width(), c.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(
        surface, c, dh.triangles.data(), dh.triangles.size(), dh.lights.data(),
        dh.lights.size(), smooth_shading_tag{}, ShadowTag{});
    cudaDeviceSynchronize();
}


#else // CPU Raytracing

template <typename ShadowTag, typename Camera>
void render_flat(memory_surface& surface, Camera c, world_geometry::data_handle dh)
{
    black_kernel(surface);
    trace_triangles_shaded(surface, c, dh.triangles, dh.lights, flat_shading_tag{},
                           ShadowTag{});
}

template <typename ShadowTag, typename Camera>
void render_smooth(memory_surface& surface, Camera c, world_geometry::data_handle dh)
{
    black_kernel(surface);
    trace_triangles_shaded(surface, c, dh.triangles, dh.lights, smooth_shading_tag{},
                           ShadowTag{});
}

#endif

#endif /* end of include guard: WORLD_SHADING_H_E0JFD9ZA */
