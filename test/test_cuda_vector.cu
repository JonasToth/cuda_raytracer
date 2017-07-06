#include "gtest/gtest.h"
#include "graphic/vector.h"

TEST(vector, construction)
{
    // default
    coord C1;
    ASSERT_EQ(C1.x, 0.);
    ASSERT_EQ(C1.y, 0.);
    ASSERT_EQ(C1.z, 0.);
    ASSERT_EQ(C1.w, 1.);

    coord C2{1., 2., 3.};
    ASSERT_EQ(C2.x, 1.);
    ASSERT_EQ(C2.y, 2.);
    ASSERT_EQ(C2.z, 3.);
    ASSERT_EQ(C2.w, 1.);
}

TEST(vector, products)
{
    coord V1{1, 0, 0}, V2{0,1,0}, V3{0,0,1};

    // DOT Product will be zero if vectors are perpendicular
    ASSERT_EQ(dot(V1,V2), 0);
    ASSERT_EQ(dot(V2,V3), 0);
    ASSERT_EQ(dot(V1,V3), 0);

    // Basic check if the result is sane
    ASSERT_EQ(dot(V1,V1), 1);
    ASSERT_EQ(dot(V2,V2), 1);
    ASSERT_EQ(dot(V3,V3), 1);

    // Crossproduct results in perpendicular vector
    auto E3 = cross(V1,V2);
    ASSERT_EQ(E3.x, 0);
    ASSERT_EQ(E3.y, 0);
    ASSERT_EQ(E3.z, 1);
}

TEST(vector, norm)
{
    coord V1{10, 0, 0}, V2{0, 10, 0}, V3{0, 0, 10}, V4{4, -2, 10};

    ASSERT_EQ(norm(V1), 10);
    ASSERT_EQ(norm(V2), 10);
    ASSERT_EQ(norm(V3), 10);
    ASSERT_NEAR(norm(V4), 10.9545, 0.001);

    ASSERT_FLOAT_EQ(normalize(V1).x, 1);
    ASSERT_FLOAT_EQ(normalize(V2).y, 1);
    ASSERT_FLOAT_EQ(normalize(V3).z, 1);
}

TEST(vector, math_operators)
{
    coord V1{15, 10, -4}, V2{-5, 12, 10};

    auto Sum = V1 + V2;
    ASSERT_EQ(Sum.x, 10);
    ASSERT_EQ(Sum.y, 22);
    ASSERT_EQ(Sum.z, 6);

    auto Difference = V1 - V2;
    ASSERT_EQ(Difference.x, 20);
    ASSERT_EQ(Difference.y, -2);
    ASSERT_EQ(Difference.z, -14);

    auto Scaled = 10. * V1;
    ASSERT_EQ(Scaled.x, 150);
    ASSERT_EQ(Scaled.y, 100);
    ASSERT_EQ(Scaled.z, -40);
}

TEST(vector, utility)
{
    const coord V1{0, 0, 0}, V2{0, 0, 0}, V3{0, 0, 0};
    ASSERT_EQ(spansArea(V1, V2, V3), false);

    const coord V4{1, 0, 0}, V5{0, 1, 0}, V6{0, 0, 1};
    ASSERT_EQ(spansArea(V4, V5, V6), true);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
