#ifndef SHADING_H_7ITEXQWS
#define SHADING_H_7ITEXQWS

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/material.h"
#include "macros.h"
#include <gsl/gsl>

struct color {
    float r;
    float g;
    float b;
};

/// Calculate ambient lighting from the global coefficient and the material coefficient
CUCALL ALWAYS_INLINE inline float ambient(float ka, float ia) noexcept { return ka * ia; }

/// Calculate diffuse lighting from material and light coeffs + Surface Normal(N), Light
/// direction (L)
CUCALL inline float diffuse(float kd, float id, float dot_product);

/// Calculate specular reflection depending on material, light, direction of camera(V)
/// and direction of reflection ray (R) and shininess alpha
CUCALL inline float specular(float ks, float is, float dot_product, float alpha);

/// Tag dispatch for different shading methods,
/// interpolated vertex normals => smooth
/// facenormal => flat
struct flat_shading_tag {
};
struct smooth_shading_tag {
};

CUCALL inline coord shading_normal(const triangle& t, coord hit,
                                   flat_shading_tag /* unused */);
CUCALL inline coord shading_normal(const triangle& t, coord hit,
                                   smooth_shading_tag /* unused */);


/// Calculate the whole shading formular for one channel
/// This is C-Style, since it must run on the gpu as well, therefor no nice vectors
template <typename ShadingStyleTag>
CUCALL color phong_shading(const phong_material* m, float ambient_constant,
                           const light_source* lights, std::size_t n_lights,
                           const coord& ray_direction, const intersect& hit,
                           ShadingStyleTag sst);


#include "shading.inl"


#endif /* end of include guard: SHADING_H_7ITEXQWS */
