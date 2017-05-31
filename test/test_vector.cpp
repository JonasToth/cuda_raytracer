#include "gtest/gtest.h"
#include "vector.h"

TEST(vector, construction)
{
    // default
    coord c1;
    ASSERT_EQ(c1.x, 0.);
    ASSERT_EQ(c1.y, 0.);
    ASSERT_EQ(c1.z, 0.);
    ASSERT_EQ(c1.w, 1.);

    coord c2{1., 2., 3.};
    ASSERT_EQ(c2.x, 1.);
    ASSERT_EQ(c2.y, 2.);
    ASSERT_EQ(c2.z, 3.);
    ASSERT_EQ(c2.w, 1.);
}

TEST(vector, products)
{
    coord v1{1, 0, 0}, v2{0,1,0}, v3{0,0,1};

    // DOT Product will be zero if vectors are perpendicular
    ASSERT_EQ(dot(v1,v2), 0);
    ASSERT_EQ(dot(v2,v3), 0);
    ASSERT_EQ(dot(v1,v3), 0);

    // Basic check if the result is sane
    ASSERT_EQ(dot(v1,v1), 1);
    ASSERT_EQ(dot(v2,v2), 1);
    ASSERT_EQ(dot(v3,v3), 1);

    // Crossproduct results in perpendicular vector
    auto e3 = cross(v1,v2);
    ASSERT_EQ(e3.x, 0);
    ASSERT_EQ(e3.y, 0);
    ASSERT_EQ(e3.z, 1);
}

TEST(vector, norm)
{
    coord v1{10, 0, 0}, v2{0, 10, 0}, v3{0, 0, 10}, v4{4, -2, 10};

    ASSERT_EQ(norm(v1), 10);
    ASSERT_EQ(norm(v2), 10);
    ASSERT_EQ(norm(v3), 10);
    ASSERT_NEAR(norm(v4), 10.9545, 0.001);

    ASSERT_FLOAT_EQ(normalize(v1).x, 1);
    ASSERT_FLOAT_EQ(normalize(v2).y, 1);
    ASSERT_FLOAT_EQ(normalize(v3).z, 1);
}

TEST(vector, math_operators)
{
    coord v1{15, 10, -4}, v2{-5, 12, 10};

    auto sum = v1 + v2;
    ASSERT_EQ(sum.x, 10);
    ASSERT_EQ(sum.y, 22);
    ASSERT_EQ(sum.z, 6);

    auto difference = v1 - v2;
    ASSERT_EQ(difference.x, 20);
    ASSERT_EQ(difference.y, -2);
    ASSERT_EQ(difference.z, -14);

    auto scaled = 10. * v1;
    ASSERT_EQ(scaled.x, 150);
    ASSERT_EQ(scaled.y, 100);
    ASSERT_EQ(scaled.z, -40);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
