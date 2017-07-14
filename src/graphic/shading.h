#ifndef SHADING_H_7ITEXQWS
#define SHADING_H_7ITEXQWS

#include "graphic/camera.h"
#include "graphic/light.h"
#include "graphic/material.h"
#include "macros.h"

struct color {
    float r;
    float g;
    float b;
};

/// Calculate ambient lighting from the global coefficient and the material coefficient
CUCALL ALWAYS_INLINE inline float ambient(float ka, float ia) noexcept { return ka * ia; }

/// Calculate diffuse lighting from material and light coeffs + Surface Normal(N), Light
/// direction (L)
CUCALL ALWAYS_INLINE inline float diffuse(float kd, float id, float dot_product)
{
    return kd * dot_product * id;
}

/// Calculate specular reflection depending on material, light, direction of camera(V)
/// and direction of reflection ray (R) and shininess alpha
CUCALL ALWAYS_INLINE inline float specular(float ks, float is, float dot_product,
                                           float alpha)
{
    return ks * std::pow(dot_product, alpha) * is;
}


// struct color { float r; float g; float b; };

/// Calculate the whole shading formular for one channel
/// This is C-Style, since it must run on the gpu as well, therefor no nice vectors
CUCALL color phong_shading(const phong_material* m, float ambient_constant,
                           const light_source* lights, std::size_t light_count,
                           const coord& ray_direction, const intersect& hit);


#include "shading.inl"


#endif /* end of include guard: SHADING_H_7ITEXQWS */
