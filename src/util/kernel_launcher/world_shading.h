#ifndef WORLD_SHADING_H_E0JFD9ZA
#define WORLD_SHADING_H_E0JFD9ZA


#include "graphic/kernels/shaded.h"
#include "graphic/kernels/utility.h"
#include "management/world.h"

#ifdef __CUDACC__
void raytrace_many_shaded(cudaSurfaceObject_t surface, world_geometry::data_handle dh);
#include "util/kernel_launcher/world_shading.inl"

#else

inline void raytrace_many_shaded(memory_surface& surface, world_geometry::data_handle dh,
                                 int)
{
    black_kernel(surface);
    trace_triangles_shaded(surface, dh.cam, dh.triangles, dh.triangle_count, dh.lights,
                           dh.light_count);
}
#endif

#endif /* end of include guard: WORLD_SHADING_H_E0JFD9ZA */
