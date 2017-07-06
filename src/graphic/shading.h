#ifndef SHADING_H_7ITEXQWS
#define SHADING_H_7ITEXQWS

#include "graphic/camera.h"
#include "graphic/material.h"
#include "graphic/light.h"
#include "macros.h"

#ifndef __CUDACC__
#   include <gsl/gsl>
#endif

/// Calculate ambient lighting from the global coefficient and the material coefficient
CUCALL ALWAYS_INLINE inline float ambient(float ka, float ia) noexcept { return ka * ia; }

/// Calculate diffuse lighting from material and light coeffs + Surface Normal(N), Light direction (L)
CUCALL ALWAYS_INLINE inline float diffuse(float kd, float id, coord N, coord L)
{
    // GSL not in Cuda, since no exceptions possible
#ifndef __CUDACC__
    Expects(norm(N) - 1.f < 0.0001f);
    Expects(norm(L) - 1.f < 0.0001f);
#endif

    return kd * dot(L, N) * id;
}

/// Calculate specular reflection depending on material, light, direction of camera(V)
/// and direction of reflection ray (R) and shininess alpha
CUCALL ALWAYS_INLINE inline float specular(float ks, float is, coord V, coord R, float alpha)
{
    return ks * std::pow(dot(R, V), alpha) * is;
    //return ks * std::pow(0.5f, alpha) * is;
}


/// Calculate the whole shading formular for one channel
/// This is C-Style, since it must run on the gpu as well, therefor no nice vectors
CUCALL float phong_shading(const phong_material& m, 
                           const light_source* lights, std::size_t light_count,
                           const coord& ray_direction, const intersect& hit);



#endif /* end of include guard: SHADING_H_7ITEXQWS */
