#ifndef MATERIAL_H_YZ53U2I4
#define MATERIAL_H_YZ53U2I4

#include "macros.h"

/// Implement the phong reflection model
struct phong_material {
    struct param {
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

    CUCALL phong_material(const float spec[3], const float diff[3], const float amb[3], 
                          float shininess)
        : r{spec[0], diff[0], amb[0]}
        , g{spec[1], diff[1], amb[1]}
        , b{spec[2], diff[2], amb[2]}
        , alpha{shininess}
    {}

    CUCALL void shininess(float s) noexcept { alpha = s; }
    CUCALL float shininess() const noexcept { return alpha; }

    param r;        ///< red channel
    param g;        ///< grenn channel
    param b;        ///< blue channel
    float alpha;    ///< shininess constant
};


#endif /* end of include guard: MATERIAL_H_YZ53U2I4 */
