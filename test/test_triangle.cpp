#include "graphic/material.h"
#include "graphic/triangle.h"
#include "gtest/gtest.h"

TEST(triangle, construction)
{
    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    ASSERT_EQ(T.p0().x, 0);
    ASSERT_EQ(T.p0().y, 0);
    ASSERT_EQ(T.p0().z, 0);

    ASSERT_EQ(T.p1().x, 1);
    ASSERT_EQ(T.p1().y, 0);
    ASSERT_EQ(T.p1().z, 0);

    ASSERT_EQ(T.p2().x, 0);
    ASSERT_EQ(T.p2().y, 1);
    ASSERT_EQ(T.p2().z, 0);

    const auto N = T.normal();
    ASSERT_EQ(N.x, 0);
    ASSERT_EQ(N.y, 0);
    ASSERT_EQ(N.z, 1);

    const auto p0_n = T.p0_normal();
    ASSERT_EQ(p0_n.x, 0);
    ASSERT_EQ(p0_n.y, 0);
    ASSERT_EQ(p0_n.z, 1);
    const auto p1_n = T.p1_normal();
    ASSERT_EQ(p1_n.x, 0);
    ASSERT_EQ(p1_n.y, 0);
    ASSERT_EQ(p1_n.z, 1);
    const auto p2_n = T.p2_normal();
    ASSERT_EQ(p2_n.x, 0);
    ASSERT_EQ(p2_n.y, 0);
    ASSERT_EQ(p2_n.z, 1);

    EXPECT_EQ(&T.p0_normal(), &T.normal());
    EXPECT_EQ(&T.p1_normal(), &T.normal());
    EXPECT_EQ(&T.p2_normal(), &T.normal());

    ASSERT_EQ(T.material(), nullptr) << "Material shall be zero";
}

TEST(triangle, validity)
{
    const coord P0{0, 0, 0}, P1{0, 0, 0}, P2{0, 0, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    const triangle T(&P0, &P1, &P2, &n);
    ASSERT_EQ(T.isValid(), false);
}

TEST(triangle, contains_point)
{
    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    // Edges and vertices seem not to match
    EXPECT_TRUE(T.contains(P0)) << "P0";
    EXPECT_TRUE(T.contains(P1)) << "P1";
    EXPECT_TRUE(T.contains(P2)) << "P2";

    EXPECT_TRUE(T.contains({0.5, 0.5, 0})) << "0.5 0.5 0";
    EXPECT_TRUE(T.contains({0.5, 0.0, 0})) << "0.5 0 0";

    EXPECT_TRUE(T.contains({0.5, 0.5, 1})) << "0.5 0.5 1";
    EXPECT_FALSE(T.contains({0.5, -0.5, 0})) << "0.5 -0.5 0";
    EXPECT_FALSE(T.contains({-0.5, -0.5, 0})) << "-0.5 -0.5 0";
    EXPECT_FALSE(T.contains({-0.5, 0.5, 0})) << "-0.5 0.5 0";

    EXPECT_FALSE(T.contains({-1, 0, 0})) << "-1 0 0";
    EXPECT_FALSE(T.contains({2, 0, 0})) << "2 0 0";
    EXPECT_FALSE(T.contains({0.5, -1, 0})) << "0.5 -1 0";
    EXPECT_FALSE(T.contains({0.5, 2, 0})) << "0.5 2 0";
    EXPECT_FALSE(T.contains({0.5, -1, 1})) << "0.5 -1 0";
    EXPECT_FALSE(T.contains({0.5, 2, -1})) << "0.5 2 0";
}

TEST(triangle, barycentric)
{
    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    // first coefficient belongs to P0
    // second coefficient belongs to P2
    // third coefficient belongs to P1
    EXPECT_EQ(T.barycentric(P0), coord(1.f, 0.f, 0.f)) << "not weighed correctly";
    EXPECT_EQ(T.barycentric(P1), coord(0.f, 0.f, 1.f)) << "not weighed correctly";
    EXPECT_EQ(T.barycentric(P2), coord(0.f, 1.f, 0.f)) << "not weighed correctly";

    EXPECT_EQ(T.barycentric(coord(0.5f, 0.5f, 0.f)), coord(0.f, 0.5f, 0.5f))
        << "not weighed correctly";
}

TEST(triangle, normal_interpolation)
{
    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    EXPECT_EQ(T.interpolated_normal(P0), n);
    EXPECT_EQ(T.interpolated_normal(P1), n);
    EXPECT_EQ(T.interpolated_normal(P2), n);

    EXPECT_EQ(T.interpolated_normal(coord(0.5f, 0.5f, 0.f)), n);

    coord n0(normalize({0.0f, 0.0f, 1.0f})), n1(normalize({-0.1f, 0.0f, 1.0f})),
        n2(normalize({0.1f, 0.0f, 1.0f}));
    T.p0_normal(&n0);
    T.p1_normal(&n1);
    T.p2_normal(&n2);

    const auto n_inter0 = T.interpolated_normal(coord(0.5f, 0.1f, 0.f));
    std::clog << "New Normal p1 = " << n1 << '\n'
              << "New Normal p2 = " << n2 << '\n'
              << "Interpolated norm = " << n_inter0 << std::endl;
    EXPECT_LT(norm(n_inter0) - 1.f, 0.0001f);
    EXPECT_LT(norm(n_inter0 - coord(0.f, 0.0f, 1.0f)), 0.1f);
}

TEST(triangle, material)
{
    float spec[3] = {0.f, 0.f, 0.f};
    float diff[3] = {0.f, 0.f, 0.f};
    float ambi[3] = {0.f, 0.f, 0.f};
    phong_material m(spec, diff, ambi, 0.f);

    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    T.material(&m);

    EXPECT_EQ(T.material(), &m) << "Material connection correctly set";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
