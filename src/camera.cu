#include "camera.h"


camera::camera(int width, int height, coord origin, coord steering)
    : __origin{origin}
    , __steering{steering}
    , __rotation{1.f, 0., 0.f,   0.f, 1.f, 0.f,   0.f, 0.f, 1.f}
    , __width{width}
    , __height{height}
{
    rotation({0.f, 0.f, 1.f}, steering, __rotation);
}

camera::camera(int width, int height)
    : camera(width, height, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f})
{}

ray camera::rayAt(int u, int v) const noexcept {
    // imeplement pinhole model
    const float focal_length = 100.f;
    const float fx = focal_length, fy = focal_length;
    const int cx = __width / 2;
    const int cy = __height / 2;

    const float x = (u - cx) / fx - (v - cy) / fy;
    const float y = (v - cy) / fy;

    // calculate the direction for camera coordinates 
    const float coeff = std::sqrt(1 + x*x + y*y) / (x*x + y*y + 1);
    const coord dir(coeff * x, coeff * y, coeff);

    // transform that direction into world coordinates (steering vector should do that)
    const coord rotated_dir(
            __rotation[0] * dir.x + __rotation[1] * dir.y + __rotation[2] * dir.z,
            __rotation[3] * dir.x + __rotation[4] * dir.y + __rotation[5] * dir.z,
            __rotation[6] * dir.x + __rotation[7] * dir.y + __rotation[8] * dir.z
    );

    return ray(__origin, rotated_dir);
}
