#include "camera.h"

CUCALL camera::camera(int width, int height, coord origin, coord steering)
    : __origin{origin}
    , __steering{steering}
    , __rotation{1.f, 0., 0.f,   0.f, 1.f, 0.f,   0.f, 0.f, 1.f}
    , __width{width}
    , __height{height}
{
    rotation({0.f, 0.f, 1.f}, steering, __rotation);
}

CUCALL camera::camera(int width, int height)
    : camera(width, height, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f})
{}
