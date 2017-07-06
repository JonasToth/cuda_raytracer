#ifndef PHONG_PARAMETERS_H_TMLRNOHG
#define PHONG_PARAMETERS_H_TMLRNOHG


#include "macros.h"

struct phong_param {
    // See https://en.wikipedia.org/wiki/Phong_reflection_model
    // for each coefficient
    float ks;     ///< specular reflection
    float kd;     ///< diffuse reflection
    float ka;     ///< ambient reflection

    CUCALL void specular_reflection(float kspec) noexcept { ks = kspec; }
    CUCALL float specular_reflection() const noexcept { return ks; }

    CUCALL void diffuse_reflection(float kdiff) noexcept { kd = kdiff; }
    CUCALL float diffuse_reflection() const noexcept { return kd; }

    CUCALL void ambient_reflection(float kamb) noexcept { ka = kamb; }
    CUCALL float ambient_reflection() const noexcept { return ka; }
};


#endif /* end of include guard: PHONG_PARAMETERS_H_TMLRNOHG */
