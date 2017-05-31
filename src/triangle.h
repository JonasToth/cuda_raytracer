#ifndef TRIANGLE_H_ESIZ1HBQ
#define TRIANGLE_H_ESIZ1HBQ


/** @file src/triangle.h
 * Implement the possibility to create triangles, that must be combined in a mesh/world.
 */

#include <array>

#include "vector.h"

/// A triangle is a set of 3 points, the order of the points defines the orientation.
class triangle {
public:
    triangle() = default;

    triangle(const triangle&) = default;
    triangle& operator=(const triangle&) = default;

    triangle(triangle&&) = default;
    triangle& operator=(triangle&&) = default;



private:
    std::array<coord, 3> points;
};


#endif /* end of include guard: TRIANGLE_H_ESIZ1HBQ */
