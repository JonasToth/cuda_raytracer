#ifndef TRIANGLE_H_ESIZ1HBQ
#define TRIANGLE_H_ESIZ1HBQ


/** @file src/triangle.h
 * Implement the possibility to create triangles, that must be combined in a mesh/world.
 */

#include <array>
#include <functional>
#include <gsl/gsl>
#include <stdexcept>

#include "macros.h"
#include "graphic/vector.h"

struct phong_material;

/// A triangle is a set of 3 __points, the order of the __points defines the orientation.
class triangle {
public:
    CUCALL triangle() = default;
    CUCALL explicit triangle(const coord* p0, const coord* p1, const coord* p2, 
                             const coord* normal) ;

    CUCALL triangle(const triangle&) = default;
    CUCALL triangle& operator=(const triangle&) = default;

    CUCALL triangle(triangle&&) = default;
    CUCALL triangle& operator=(triangle&&) = default;

    CUCALL ~triangle() = default;

    CUCALL const coord& p0() const noexcept { return *__points[0]; }
    CUCALL const coord& p0_normal() const noexcept { return *__normals[0]; }
    CUCALL void p0_normal(const coord* p0_n) noexcept { __normals[0] = p0_n; }

    CUCALL const coord& p1() const noexcept { return *__points[1]; }
    CUCALL const coord& p1_normal() const noexcept { return *__normals[1]; }
    CUCALL void p1_normal(const coord* p1_n) noexcept { __normals[1] = p1_n; }

    CUCALL const coord& p2() const noexcept { return *__points[2]; }
    CUCALL const coord& p2_normal() const noexcept { return *__normals[2]; }
    CUCALL void p2_normal(const coord* p2_n) noexcept { __normals[2] = p2_n; }

    CUCALL void material(const phong_material* m) noexcept { __material = m; }
    CUCALL const phong_material* material() const noexcept { return __material; }

    CUCALL void normal(const coord* n) noexcept { __normal = n; }
    /// Surface normal of the triangle, either returned from normal pool or calculated
    CUCALL const coord& normal() const noexcept { return *__normal;  }

    CUCALL bool contains(const coord P) const noexcept;

    // expects: Triangle contains P
    // 
    // first coefficient belongs to P0
    // second coefficient belongs to P2
    // third coefficient belongs to P1
    CUCALL coord barycentric(coord P) const noexcept;

    CUCALL coord interpolated_normal(coord P) const noexcept;

    CUCALL bool isValid() const noexcept;

private:
    const coord* __points[3];         ///< optimization, triangles can share vertices
    const coord* __normals[3];        ///< vertex normals
    const coord* __normal;            ///< triangle normal 
    const phong_material* __material; ///< triangles share materials
};

#include "graphic/triangle.inl"

#endif /* end of include guard: TRIANGLE_H_ESIZ1HBQ */
