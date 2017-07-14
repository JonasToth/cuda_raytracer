#ifndef WORLD_DEPTH_H_OJHV0R12
#define WORLD_DEPTH_H_OJHV0R12

#include "graphic/kernels/utility.h"
#include "graphic/kernels/trace.h"

void raytrace_many_cuda(cudaSurfaceObject_t Surface, const camera& c,
                        const triangle* Triangles, int TriangleCount);
void raytrace_cuda(cudaSurfaceObject_t& Surface, int width, int height, const triangle* T);


#include "util/kernel_launcher/world_depth.inl"

#endif /* end of include guard: WORLD_DEPTH_H_OJHV0R12 */
