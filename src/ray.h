#ifndef RAY_H_COAY6AFS
#define RAY_H_COAY6AFS

#include <utility>
#include <thrust/pair.h>
#include <thrust/tuple.h>

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
    ray() = default;
    ray(const coord& origin, const coord& direction) 
        : origin{origin}
        , direction{normalize(direction)} {}
    ray(const ray&) = default;
    ray(ray&&) = default;
    ray& operator=(const ray&) = default;
    ray& operator=(ray&&) = default;
    ~ray() = default;


    LIB::pair<bool, intersect> intersects(const triangle& Tri) const noexcept {
        const auto TNormal = Tri.normal();

        // plane equation: Ax + By + Cz + D = 0, compute D
        const float D = dot(TNormal, Tri.p0());

        // https://www.scratchapixel.com/lessons/3d-basic-rendering/
        // ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution

        // ray equation: P = O + t * R, solve for t and P
        //std::cout << TNormal.x << " " << TNormal.y << " " << TNormal.z << std::endl;
        //std::cout << direction.x << " " << direction.y << " " << direction.z 
                  //<< std::endl << std::endl;
        float Divisor = dot(TNormal, direction);

        // rays are parallel, so no intersection
        if(Divisor < 0.0001 && Divisor > -0.0001)
            return LIB::make_pair(false, intersect{});
        
        const float T = (dot(TNormal, origin) + D) / Divisor;
        const coord Hit = origin + T * direction;

        // calculate the hit normal
        const coord HitNormal(Hit);

        if(T > 0.f)
            return LIB::make_pair(true, intersect{T, Hit, HitNormal});
        else
            return LIB::make_pair(false, intersect{T, Hit, HitNormal});
    }

    coord origin;
    coord direction;
};

#endif /* end of include guard: RAY_H_COAY6AFS */
