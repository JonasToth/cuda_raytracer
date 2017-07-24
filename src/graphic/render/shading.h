#ifndef WORLD_SHADING_H_E0JFD9ZA
#define WORLD_SHADING_H_E0JFD9ZA


#include "graphic/kernels/shaded.h"
#include "graphic/kernels/utility.h"
#include "management/world.h"

#ifdef __CUDACC__ // GPU Raytracing
inline void render_flat(cudaSurfaceObject_t surface, world_geometry::data_handle dh)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((dh.cam.width() + dimBlock.x) / dimBlock.x,
                 (dh.cam.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, dh.cam.width(), dh.cam.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(
        surface, dh.cam, dh.triangles.data(), dh.triangles.size(), dh.lights.data(),
        dh.lights.size(), flat_shading_tag{}, hard_shadow_tag{});
    cudaDeviceSynchronize();
}
inline void render_smooth(cudaSurfaceObject_t surface, world_geometry::data_handle dh)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((dh.cam.width() + dimBlock.x) / dimBlock.x,
                 (dh.cam.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, dh.cam.width(), dh.cam.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(
        surface, dh.cam, dh.triangles.data(), dh.triangles.size(), dh.lights.data(),
        dh.lights.size(), smooth_shading_tag{}, hard_shadow_tag{});
    cudaDeviceSynchronize();
}


#else // CPU Raytracing

inline void render_flat(memory_surface& surface, world_geometry::data_handle dh)
{
    black_kernel(surface);
    trace_triangles_shaded(surface, dh.cam, dh.triangles, dh.lights, flat_shading_tag{},
                           hard_shadow_tag{});
}

inline void render_smooth(memory_surface& surface, world_geometry::data_handle dh)
{
    black_kernel(surface);
    trace_triangles_shaded(surface, dh.cam, dh.triangles, dh.lights, smooth_shading_tag{},
                           hard_shadow_tag{});
}

#endif

#endif /* end of include guard: WORLD_SHADING_H_E0JFD9ZA */
