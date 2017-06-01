#ifndef TRIANGLE_H_ESIZ1HBQ
#define TRIANGLE_H_ESIZ1HBQ


/** @file src/triangle.h
 * Implement the possibility to create triangles, that must be combined in a mesh/world.
 */

#include <array>
#include <functional>
#include <stdexcept>

#include "macros.h"
#include "vector.h"

/// A triangle is a set of 3 __points, the order of the __points defines the orientation.
class triangle {
public:
    CUCALL triangle(const coord& p0, const coord& p1, const coord& p2) : __points{p0, p1, p2} {}

    CUCALL triangle(const triangle&) = default;
    CUCALL triangle& operator=(const triangle&) = default;

    CUCALL triangle(triangle&&) = default;
    CUCALL triangle& operator=(triangle&&) = default;

    CUCALL const coord& p0() const noexcept { return __points[0]; }
    CUCALL const coord& p1() const noexcept { return __points[1]; }
    CUCALL const coord& p2() const noexcept { return __points[2]; }

    /// Surface normal of the triangle, not normalized
    CUCALL coord normal() const noexcept { return cross(p1() - p0(), p2() - p1()); }

    bool isValid() const noexcept { return spansArea(__points[0], __points[1], __points[2]); }

private:
    std::array<std::reference_wrapper<const coord>, 3> __points;
};


#endif /* end of include guard: TRIANGLE_H_ESIZ1HBQ */
