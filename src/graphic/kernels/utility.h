#ifndef UTILITY_H_SZFM0XUL
#define UTILITY_H_SZFM0XUL


#ifdef __CUDACC__

#include "management/surface_raii.h"
__global__ void black_kernel(cudaSurfaceObject_t surface, int width, int height);
__global__ void stupid_colors(cudaSurfaceObject_t surface, int width, int height, float t);
#include "graphic/kernels/utility.inl"

#else

#include "management/memory_surface.h"
void stupid_colors(memory_surface& surface, float t);
void black_kernel(memory_surface& surface);
#endif

#endif /* end of include guard: UTILITY_H_SZFM0XUL */
