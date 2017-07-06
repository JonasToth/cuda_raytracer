#ifndef SHADED_H_KK3H1DCZ
#define SHADED_H_KK3H1DCZ


#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/triangle.h"
#include "graphic/ray.h"
#include "graphic/shading.h"
#include "management/surface_raii.h"


__global__ void trace_many_triangles_shaded(cudaSurfaceObject_t surface, camera c,
                                            const triangle* triangles, std::size_t n_triangles,
                                            const light_source* lights, std::size_t n_lights,
                                            int width, int height);

#endif /* end of include guard: SHADED_H_KK3H1DCZ */
