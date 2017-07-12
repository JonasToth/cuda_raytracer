#ifndef RAY_H_COAY6AFS
#define RAY_H_COAY6AFS

#include <utility>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include "macros.h"
#include "graphic/triangle.h"
#include "graphic/vector.h"

struct intersect {
    CUCALL intersect() = default;
    CUCALL explicit intersect(float depth, const coord hit, const coord n) 
        : depth{depth}
        , hit{hit}
        , normal{n} {}

    CUCALL intersect(const intersect&) = default;
    CUCALL intersect(intersect&&) = default;

    CUCALL intersect& operator=(const intersect&) = default;
    CUCALL intersect& operator=(intersect&&) = default;

    CUCALL ~intersect() = default;

    float depth = 0.f;
    coord hit;
    coord normal;
};

struct ray {
    CUCALL ray() = default;
    CUCALL explicit ray(const coord origin, const coord direction) 
        : origin{origin}
        , direction{normalize(direction)} {}

    CUCALL ray(const ray&) = default;
    CUCALL ray(ray&&) = default;

    CUCALL ray& operator=(const ray&) = default;
    CUCALL ray& operator=(ray&&) = default;

    CUCALL ~ray() = default;

    /// Calculate if the ray truly intersects the triangle, and give intersection information.
    CUCALL LIB::pair<bool, intersect> intersects(const triangle& Tri) const noexcept {
        const auto TNormal = Tri.normal();

        // https://www.scratchapixel.com/lessons/3d-basic-rendering/
        // ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution

        //std::cout << TNormal.x << " " << TNormal.y << " " << TNormal.z << std::endl;
        //std::cout << direction.x << " " << direction.y << " " << direction.z 
                  //<< std::endl << std::endl;
                  
        // ray equation: P = O + t * R, solve for t and P
        float Divisor = dot(TNormal, direction);

        // rays are parallel, so no intersection
        if(Divisor == 0.f)
            return LIB::make_pair(false, intersect{});

        const float T = dot(TNormal, Tri.p0() - origin) / Divisor;
        const coord Hit = origin + T * direction;

        if(T > 0.f)
            return LIB::make_pair(Tri.contains(Hit), intersect{T, Hit, TNormal});
        else
            return LIB::make_pair(false, intersect{T, Hit, TNormal});
    }

    coord origin;
    coord direction;
};

#endif /* end of include guard: RAY_H_COAY6AFS */
