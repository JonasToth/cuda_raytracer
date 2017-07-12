#include "gtest/gtest.h"
#include "graphic/triangle.h"
#include "graphic/material.h"

TEST(triangle_test, construction)
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

    ASSERT_EQ(T.material(), nullptr) << "Material shall be zero";
}

TEST(triangle_test, validity)
{
    const coord P0{0, 0, 0}, P1{0, 0, 0}, P2{0, 0, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    const triangle T(&P0, &P1, &P2, &n);
    ASSERT_EQ(T.isValid(), false);
}

TEST(triangle_test, contains_point)
{
    const coord P0{0, 0, 0}, P1{1, 0, 0}, P2{0, 1, 0};
    const coord n = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &n};

    // Edges and vertices seem not to match
    EXPECT_EQ(T.contains(P0), true) << "P0";
    EXPECT_EQ(T.contains(P1), true) << "P1";
    EXPECT_EQ(T.contains(P2), true) << "P2";

    EXPECT_EQ(T.contains({0.5, 0.5, 0}), true) << "0.5 0.5 0";
    EXPECT_EQ(T.contains({0.5, 0.0, 0}), true) << "0.5 0 0";

    EXPECT_EQ(T.contains({0.5, 0.5, 1}), true) << "0.5 0.5 1";
    EXPECT_EQ(T.contains({0.5, -0.5, 0}), false) << "0.5 -0.5 0";
    EXPECT_EQ(T.contains({-0.5, -0.5, 0}), false) << "-0.5 -0.5 0";
    EXPECT_EQ(T.contains({-0.5, 0.5, 0}), false) << "-0.5 0.5 0";

    EXPECT_EQ(T.contains({-1, 0, 0}), false) << "-1 0 0";
    EXPECT_EQ(T.contains({2, 0, 0}), false) << "2 0 0";
    EXPECT_EQ(T.contains({0.5, -1, 0}), false) << "0.5 -1 0";
    EXPECT_EQ(T.contains({0.5, 2, 0}), false) << "0.5 2 0";
}

TEST(triangle_test, material)
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
