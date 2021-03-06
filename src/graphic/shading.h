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

/// Tag dispatch to create shadows, either no shadow or sharp shadow
struct no_shadow_tag {
};
struct hard_shadow_tag {
};

/// Returns face normal of the triangle
CUCALL inline coord shading_normal(const triangle& t, coord hit,
                                   flat_shading_tag /* unused */);
/// Returns interpolated normal of the hit point on the triangle
CUCALL inline coord shading_normal(const triangle& t, coord hit,
                                   smooth_shading_tag /* unused */);


/// Calculate the whole shading formular for one channel
/// This is C-Style, since it must run on the gpu as well, therefor no nice vectors
template <typename ShadingStyleTag, typename ShadowTag>
CUCALL color phong_shading(const phong_material* m, float ambient_constant,
                           const coord& ray_direction, const intersect& hit,
                           gsl::span<const light_source> lights,
                           gsl::span<const triangle> triangles, ShadingStyleTag sst,
                           ShadowTag st);

/// Returns always true, since no shadows shall be calculated
CUCALL ALWAYS_INLINE inline bool luminated_by_light(const intersect& /*unused*/,
                                                    const light_source& /*unused*/,
                                                    gsl::span<const triangle> /*unused*/,
                                                    no_shadow_tag /*unused*/)
{
    return true;
}

/// Test if there is a triangle between the intersection point and the light source l.
CUCALL inline bool luminated_by_light(const intersect& hit, const light_source& l,
                                      gsl::span<const triangle> triangles,
                                      hard_shadow_tag /*unused*/);

#include "shading.inl"


#endif /* end of include guard: SHADING_H_7ITEXQWS */
