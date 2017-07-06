#ifndef MATERIAL_H_YZ53U2I4
#define MATERIAL_H_YZ53U2I4

#include "macros.h"

/// Implement the phong reflection model
struct phong_material {
    // See https://en.wikipedia.org/wiki/Phong_reflection_model
    // for each coefficient
    float ks;     ///< specular reflection
    float kd;     ///< diffuse reflection
    float ka;     ///< ambient reflection
    float alpha;  ///< shininess constant

    CUCALL void specular_reflection(float kspec) noexcept { ks = kspec; }
    CUCALL float specular_reflection() const noexcept { return ks; }

    CUCALL void diffuse_reflection(float kdiff) noexcept { kd = kdiff; }
    CUCALL float diffuse_reflection() const noexcept { return kd; }

    CUCALL void ambient_reflection(float kamb) noexcept { ka = kamb; }
    CUCALL float ambient_reflection() const noexcept { return ka; }

    CUCALL void shininess(float s) noexcept { alpha = s; }
    CUCALL float shininess() const noexcept { return alpha; }
};


#endif /* end of include guard: MATERIAL_H_YZ53U2I4 */
