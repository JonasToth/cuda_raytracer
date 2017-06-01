#ifndef RAY_H_COAY6AFS
#define RAY_H_COAY6AFS

#include <utility>
#include <thrust/pair.h>

#include "macros.h"
#include "triangle.h"
#include "vector.h"

struct intersect {
    CUCALL intersect() = default;
    CUCALL intersect(float depth, const coord& hit, const coord& n) 
        : depth{depth}
        , hit{hit}
        , normal{n} {}

    float depth = 0.f;
    coord hit;
    coord normal;
};

struct ray {
    LIB::pair<bool, intersect> intersects(const triangle& tri) const noexcept {
        const auto t_normal = tri.normal();

        // plane equation: Ax + By + Cz + D = 0, compute D
        const float D = dot(t_normal, tri.p0());

        // ray equation: P = O + t * R, solve for t and P
        float divisor = dot(t_normal, direction);

        // rays are parallel, so no intersection
        if(std::abs(divisor) < 0.0001)
            return LIB::make_pair(false, intersect{});
        
        const float t = -(dot(t_normal, origin) + D) / divisor;
        const coord hit = origin + t * direction;

        const coord hit_normal(hit);

        if(t > 0.f)
            return LIB::make_pair(true, intersect{t, hit, hit_normal});
        else
            return LIB::make_pair(false, intersect{t, hit, hit_normal});
    }

    coord origin;
    coord direction;
};

#endif /* end of include guard: RAY_H_COAY6AFS */
