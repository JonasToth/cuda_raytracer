#ifndef SHADING_H_7ITEXQWS
#define SHADING_H_7ITEXQWS

#include "graphic/material.h"
#include "graphic/light.h"
#include "macros.h"

CUCALL ALWAYS_INLINE inline float ambient(float ka, float ia) noexcept { return ka * ia; }


#endif /* end of include guard: SHADING_H_7ITEXQWS */
