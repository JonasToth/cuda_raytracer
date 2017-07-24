
CUCALL inline triangle::triangle(const coord* p0, const coord* p1, const coord* p2,
                                 const coord* normal)
  : __points{p0, p1, p2}
  , __normals{normal, normal, normal}
  , __normal{normal}
  , __material{nullptr}
{
}

CUCALL inline bool triangle::contains(const coord P) const noexcept
{
    const auto E0 = p1() - p0();
    const auto E1 = p2() - p1();
    const auto E2 = p0() - p2();

    const auto C0 = P - p0();
    const auto C1 = P - p1();
    const auto C2 = P - p2();

    const auto N = cross(E0, E1);

    return dot(N, cross(E0, C0)) >= 0 && dot(N, cross(E1, C1)) >= 0 &&
           dot(N, cross(E2, C2)) >= 0;
}

CUCALL inline coord triangle::barycentric(const coord P) const noexcept
{
#ifndef __CUDACC__
    Expects(contains(P));
#endif
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    // https://computergraphics.stackexchange.com/questions/5006/how-do-i-use-barycentric-coordinates-to-interpolate-vertex-normal
    // the version differ, i rotated the difference vectors a bit
    const coord v0 = p0() - p1(), v1 = p2() - p1(), v2 = P - p1();
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float den = 1.f / (d00 * d11 - d01 * d01);

    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);

    const float v = den * (d11 * d20 - d01 * d21), w = den * (d00 * d21 - d01 * d20),
                u = 1.f - v - w;
    const coord bary(v, w, u);


#ifndef __CUDACC__
    Ensures(v >= -0.01f && v <= 1.01f);
    Ensures(w >= -0.01f && w <= 1.01f);
    Ensures(u >= -0.01f && u <= 1.01f);
    Ensures(norm(bary) <= 1.f);
#endif

    return bary;
}

CUCALL inline coord triangle::interpolated_normal(const coord P) const noexcept
{
#ifndef __CUDACC__
    Expects(contains(P));
#endif
    const coord bary = barycentric(P);
    // Coefficients belong to specific vertices
    // first coeff  -> P0
    // second coeff -> P2
    // third coeff  -> P1
    const auto intp_n =
        normalize(bary.x * p0_normal() + bary.y * p2_normal() + bary.z * p1_normal());

#ifndef __CUDACC__
    Ensures(std::fabs(norm(intp_n) - 1.f) < 0.00001f);
#endif
    return intp_n;
}

CUCALL inline bool triangle::isValid() const noexcept
{
    return spansArea(*__points[0], *__points[1], *__points[2]);
}
