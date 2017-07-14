#include "util/kernel_launcher/world_shading.h"

void raytrace_many_shaded(cudaSurfaceObject_t surface, camera c,
                                 const triangle* triangles, std::size_t n_triangles,
                                 const light_source* lights, std::size_t n_lights)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, c.width(), c.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(surface, c,
                                                  triangles, n_triangles, 
                                                  lights, n_lights,
                                                  c.width(), c.height());
}


