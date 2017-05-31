#ifndef VECTOR_H_SWTQE942
#define VECTOR_H_SWTQE942

#include <cmath>

/// Implements 3d homogenous coordinates.
struct coord
{
    /// Constructor will 3d coordinates
    coord(float x, float y, float z) : x{x}, y{y}, z{z} {}

    coord() = default;

    coord(const coord&) = default;
    coord& operator=(const coord&) = default;

    coord(coord&&) = default;
    coord& operator=(coord&&) = default;


    float x = 0;
    float y = 0;
    float z = 0;
    float w = 1;
};

// ===== MATH FUNCTIONS =====

/// Euclid norm.
float norm(const coord& v) noexcept { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }

/// Normalize the vector.
coord normalize(const coord& v) noexcept {
    auto n = norm(v);
    return coord{
        v.x / n,
        v.y / n,
        v.z / n
    };
}

/// Dot product of 2 vectors 3d.
float dot(const coord& lhs, const coord& rhs) noexcept { 
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/// Implements cross product.
coord cross(const coord& lhs, const coord& rhs) noexcept {
    return coord{
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}


/// Vector subtraction
coord operator-(const coord& lhs, const coord& rhs) noexcept {
    return coord{
        lhs.x - rhs.x,
        lhs.y - rhs.y,
        lhs.z - rhs.z
    };
}

/// Vector addition
coord operator+(const coord& lhs, const coord& rhs) noexcept {
    return coord{
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z
    };
}

/// Scalar - Vector multiplication
coord operator*(float s, const coord& rhs) noexcept {
    return coord{
        s * rhs.x,
        s * rhs.y,
        s * rhs.z,
    };
}


#endif /* end of include guard: VECTOR_H_SWTQE942 */
