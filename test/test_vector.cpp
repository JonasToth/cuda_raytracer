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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
