#ifndef CAMERA_H_5RCCQYDN
#define CAMERA_H_5RCCQYDN

#include "vector.h"
#include "ray.h"

class camera {
public:
    CUCALL camera(int width, int height);
    CUCALL camera(int width, int height, coord origin, coord steering);

    CUCALL ray rayAt(int u, int v) const noexcept;

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


#endif /* end of include guard: CAMERA_H_5RCCQYDN */
