#ifndef CAMERA_H_5RCCQYDN
#define CAMERA_H_5RCCQYDN

#include "vector.h"
#include "ray.h"

class camera {
public:
    CUCALL camera(int width, int height);
    CUCALL camera(int width, int height, coord origin, coord steering);

    CUCALL ray rayAt(int u, int v) const noexcept;

    CUCALL void move(coord translation) noexcept { __origin = __origin + translation; }

    CUCALL int width() const noexcept { return __width; }
    CUCALL int height() const noexcept { return __height; }

    CUCALL coord origin() const noexcept { return __origin; }
    CUCALL coord steering() const noexcept { return __steering; }

private:
    coord __origin;
    coord __steering;

    float __rotation[9];

    int __width;
    int __height;
};

inline ray camera::rayAt(int u, int v) const noexcept {
    // imeplement pinhole model
    const float focal_length = 360.f;
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

#endif /* end of include guard: CAMERA_H_5RCCQYDN */
