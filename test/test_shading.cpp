#include "gtest/gtest.h"

#include "graphic/shading.h"


TEST(shading, ambient_coeff)
{
    EXPECT_FLOAT_EQ(ambient(0.f, 0.f), 0.f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(0.f, 1.f), 0.f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(1.f, 0.f), 0.f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(1.f, 1.f), 1.f) << "Expected ambient 1";
}

TEST(shading, diffuse_coeff)
{

}

TEST(shading, specular_coeff)
{

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
