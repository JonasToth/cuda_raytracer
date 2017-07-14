#ifndef LIGHT_H_RWS2MXT5
#define LIGHT_H_RWS2MXT5

#include "graphic/phong_parameters.h"
#include "graphic/vector.h"

struct phong_light {
    CUCALL phong_light() = default;
    CUCALL explicit phong_light(const float spec[3], const float diff[3])
      : r{spec[0], diff[0]}
      , g{spec[1], diff[1]}
      , b{spec[2], diff[2]}
    {
    }
    CUCALL phong_light(const phong_light&) = default;
    CUCALL phong_light& operator=(const phong_light&) = default;

    phong_param_light r;
    phong_param_light g;
    phong_param_light b;
};


struct light_source {
    CUCALL light_source() = default;
    CUCALL explicit light_source(phong_light l, coord p)
      : light{l}
      , position{p}
    {
    }

    CUCALL light_source(const light_source&) = default;
    CUCALL light_source& operator=(const light_source&) = default;

    phong_light light;
    coord position;
};


#endif /* end of include guard: LIGHT_H_RWS2MXT5 */
