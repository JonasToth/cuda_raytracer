#ifndef LIGHT_H_RWS2MXT5
#define LIGHT_H_RWS2MXT5

#include "graphic/phong_parameters.h"
#include "graphic/vector.h"

struct phong_light {
    CUCALL phong_light(const float spec[3], const float diff[3], const float amb[3])
        : r{spec[0], diff[0], amb[0]}
        , g{spec[1], diff[1], amb[1]}
        , b{spec[2], diff[2], amb[2]}
    {}

    phong_param r;
    phong_param g;
    phong_param b;
};


struct light_source {
    phong_light light;
    coord position;
};


#endif /* end of include guard: LIGHT_H_RWS2MXT5 */
