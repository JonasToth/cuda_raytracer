#ifndef RAY_H_COAY6AFS
#define RAY_H_COAY6AFS

#include <utility>

#include "triangle.h"
#include "vector.h"

struct intersect {
    float depth = 0.f;
    coord hit;
    coord normal;
};

struct ray {
    std::pair<bool, intersect> intersects(const triangle& t) const noexcept {
        const auto t_normal = t.normal();

        // plane equation: Ax + By + Cz + D = 0, compute D
        const float D = dot(t_normal, t.p0());

        // ray equation: P = O + t * R, solve for t and P
        float divisor = dot(t_normal, direction);

        // rays are parallel, so no intersection
        if(std::abs(divisor) < 0.0001)
            return {false, {}};
        
        const float t = -(dot(t_normal, origin) + D) / divisor;
        const coord hit = origin + t * direction;

        if(t > 0.f)
            return {true, {t, hit, hit_normal}};
        else
            return {false, {t, hit, hit_normal}};
    }

    coord origin;
    coord direction;
};

#endif /* end of include guard: RAY_H_COAY6AFS */
