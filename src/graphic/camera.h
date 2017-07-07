#ifndef CAMERA_H_5RCCQYDN
#define CAMERA_H_5RCCQYDN

#include "graphic/vector.h"
#include "graphic/ray.h"

class camera {
public:
    CUCALL camera(int width, int height);
    CUCALL camera(int width, int height, coord origin, coord steering);

    CUCALL ray rayAt(int u, int v) const noexcept;

    CUCALL void move(coord translation) noexcept { __origin = __origin + translation; }
    CUCALL void swipe(float alpha, float beta, float gamma) noexcept;
    CUCALL void lookAt(coord target) noexcept;

    CUCALL int width() const noexcept { return __width; }
    CUCALL int height() const noexcept { return __height; }

    CUCALL coord origin() const noexcept { return __origin; }
    CUCALL coord steering() const noexcept { return __steering; }

private:
    coord __origin;
    coord __steering;

    int __width;
    int __height;
};

#endif /* end of include guard: CAMERA_H_5RCCQYDN */
