#ifndef UTILITY_H_SZFM0XUL
#define UTILITY_H_SZFM0XUL

#include "management/surface_raii.h"
#include "management/memory_surface.h"

__global__ void black_kernel(cudaSurfaceObject_t surface, int width, int height);
void black_kernel(memory_surface& surface);

__global__ void stupid_colors(cudaSurfaceObject_t surface, int width, int height, float t);
void stupid_colors(memory_surface& surface, float t);

#ifdef __CUDACC__
#include "graphic/kernels/utility.inl"
#endif

#endif /* end of include guard: UTILITY_H_SZFM0XUL */
