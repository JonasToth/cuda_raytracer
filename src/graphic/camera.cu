#include "camera.h"

CUCALL camera::camera(int width, int height, coord origin, coord steering)
    : __origin{origin}
    , __steering{normalize(steering)}
    , __width{width}
    , __height{height}
{}

CUCALL camera::camera(int width, int height)
    : camera(width, height, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f})
{}
