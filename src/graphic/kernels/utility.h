#ifndef UTILITY_H_SZFM0XUL
#define UTILITY_H_SZFM0XUL

#include "management/surface_raii.h"

__global__ void black_kernel(cudaSurfaceObject_t surface, int width, int height);
__global__ void stupid_colors(cudaSurfaceObject_t surface, int width, int height, float t);

#endif /* end of include guard: UTILITY_H_SZFM0XUL */
