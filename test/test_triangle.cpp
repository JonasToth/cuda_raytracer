#include "gtest/gtest.h"
#include "triangle.h"

TEST(triangle_test, construction)
{
    const coord p0{0, 0, 0}, p1{1, 0, 0}, p2{0, 1, 0};
    triangle t{p0, p1, p2};
    ASSERT_EQ(p0.x, 0);
    ASSERT_EQ(p0.y, 0);
    ASSERT_EQ(p0.z, 0);

    ASSERT_EQ(p1.x, 1);
    ASSERT_EQ(p1.y, 0);
    ASSERT_EQ(p1.z, 0);

    ASSERT_EQ(p2.x, 0);
    ASSERT_EQ(p2.y, 1);
    ASSERT_EQ(p2.z, 0);

    const auto n = t.normal();
    ASSERT_EQ(n.x, 0);
    ASSERT_EQ(n.y, 0);
    ASSERT_EQ(n.z, 1);
}

TEST(triangle_test, validity)
{
    const coord p0{0, 0, 0}, p1{0, 0, 0}, p2{0, 0, 0};
    ASSERT_THROW(triangle(p0, p1, p2), std::invalid_argument);
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
