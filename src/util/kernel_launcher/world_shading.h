#ifndef WORLD_SHADING_H_E0JFD9ZA
#define WORLD_SHADING_H_E0JFD9ZA


#include "graphic/kernels/utility.h"
#include "graphic/kernels/shaded.h"
#include "management/world.h"

void raytrace_many_shaded(cudaSurfaceObject_t surface, camera c,
                          const triangle* triangles, std::size_t n_triangles,
                          const light_source* lights, std::size_t n_lights);

#endif /* end of include guard: WORLD_SHADING_H_E0JFD9ZA */
