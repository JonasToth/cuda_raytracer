#ifndef MATERIAL_H_YZ53U2I4
#define MATERIAL_H_YZ53U2I4

#include "graphic/phong_parameters.h"

/// Implement the phong reflection model
struct phong_material {
    
    CUCALL explicit phong_material(const float spec[3], const float diff[3], const float amb[3], 
                                   float shininess)
        : r{spec[0], diff[0], amb[0]}
        , g{spec[1], diff[1], amb[1]}
        , b{spec[2], diff[2], amb[2]}
        , alpha{shininess}
    {}

    CUCALL void shininess(float s) noexcept { alpha = s; }
    CUCALL float shininess() const noexcept { return alpha; }

    phong_param r;        ///< red channel
    phong_param g;        ///< grenn channel
    phong_param b;        ///< blue channel
    float alpha;    ///< shininess constant
};


#endif /* end of include guard: MATERIAL_H_YZ53U2I4 */
