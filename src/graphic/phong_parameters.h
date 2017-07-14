#ifndef PHONG_PARAMETERS_H_TMLRNOHG
#define PHONG_PARAMETERS_H_TMLRNOHG


#include "macros.h"

struct phong_param_material {
    // See https://en.wikipedia.org/wiki/Phong_reflection_model
    // for each coefficient
    float ks; ///< specular reflection
    float kd; ///< diffuse reflection
    float ka; ///< ambient reflection

    CUCALL phong_param_material() = default;
    CUCALL explicit phong_param_material(float s, float d, float a)
      : ks{s}
      , kd{d}
      , ka{a}
    {
    }
    CUCALL phong_param_material(const phong_param_material&) = default;
    CUCALL phong_param_material& operator=(const phong_param_material&) = default;

    CUCALL void specular_reflection(float kspec) noexcept { ks = kspec; }
    CUCALL float specular_reflection() const noexcept { return ks; }

    CUCALL void diffuse_reflection(float kdiff) noexcept { kd = kdiff; }
    CUCALL float diffuse_reflection() const noexcept { return kd; }

    CUCALL void ambient_reflection(float kamb) noexcept { ka = kamb; }
    CUCALL float ambient_reflection() const noexcept { return ka; }
};

struct phong_param_light {
    // See https://en.wikipedia.org/wiki/Phong_reflection_model
    // for each coefficient
    float ks; ///< specular reflection
    float kd; ///< diffuse reflection

    CUCALL phong_param_light() = default;
    CUCALL explicit phong_param_light(float s, float d)
      : ks{s}
      , kd{d}
    {
    }
    CUCALL phong_param_light(const phong_param_light&) = default;
    CUCALL phong_param_light& operator=(const phong_param_light&) = default;

    CUCALL void specular_color(float kspec) noexcept { ks = kspec; }
    CUCALL float specular_color() const noexcept { return ks; }

    CUCALL void diffuse_color(float kdiff) noexcept { kd = kdiff; }
    CUCALL float diffuse_color() const noexcept { return kd; }
};


#endif /* end of include guard: PHONG_PARAMETERS_H_TMLRNOHG */
