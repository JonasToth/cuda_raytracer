#ifndef VECTOR_H_SWTQE942
#define VECTOR_H_SWTQE942

#include <cmath>
#include <iostream>

#include "macros.h"

/// Implements 3d homogenous coordinates.
struct coord
{
    /// Constructor will 3d coordinates
    CUCALL coord(float x, float y, float z) : x{x}, y{y}, z{z}, w{1} {}

    CUCALL coord() : x{0}, y{0}, z{0}, w{1} {}

    CUCALL coord(const coord&) = default;
    CUCALL coord& operator=(const coord&) = default;

    CUCALL coord(coord&&) = default;
    CUCALL coord& operator=(coord&&) = default;

    CUCALL ~coord() = default;

    float x = 0;
    float y = 0;
    float z = 0;
    float w = 1;
};

// ===== MATH FUNCTIONS =====

/// Euclid norm.
inline CUCALL float norm(const coord v) noexcept { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }

/// Normalize the vector.
inline CUCALL coord normalize(const coord v) noexcept {
    auto N = norm(v);
    return coord{
        v.x / N,
        v.y / N,
        v.z / N
    };
}

/// Dot product of 2 vectors 3d.
inline CUCALL float dot(const coord lhs, const coord rhs) noexcept { 
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/// Implements cross product.
inline CUCALL coord cross(const coord lhs, const coord rhs) noexcept {
    return coord{
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}


/// Vector subtraction
inline CUCALL coord operator-(const coord lhs, const coord rhs) noexcept {
    return coord{
        lhs.x - rhs.x,
        lhs.y - rhs.y,
        lhs.z - rhs.z
    };
}

/// Vector addition
inline CUCALL coord operator+(const coord lhs, const coord rhs) noexcept {
    return coord{
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z
    };
}

/// Scalar - Vector multiplication
inline CUCALL coord operator*(float s, const coord rhs) noexcept {
    return coord{
        s * rhs.x,
        s * rhs.y,
        s * rhs.z,
    };
}


/// Compare for equality of two coordinates, with epsilon
inline CUCALL bool operator==(const coord c1, const coord c2) noexcept
{
    return norm(c2 - c1) < 0.00001;
}

inline CUCALL bool operator!=(const coord c1, const coord c2) noexcept { return !(c1 == c2); }


/// Three __points can span an area, if they are all different
inline CUCALL bool spansArea(const coord p0, const coord p1, const coord p2) noexcept {
    return (p0 != p1) && (p0 != p2) && (p1 != p2);
}


inline std::ostream& operator<<(std::ostream& os, const coord& c)
{
    os << '(' << c.x << ',' << c.y << ',' << c.z << ',' << c.w << ')';
    return os;
}

#endif /* end of include guard: VECTOR_H_SWTQE942 */
