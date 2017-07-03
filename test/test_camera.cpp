#include "gtest/gtest.h"

#include "camera.h"

TEST(camera, properties)
{
    {
    camera c(640, 480);

    EXPECT_EQ(c.width(), 640) << "Bad width";
    EXPECT_EQ(c.height(), 480) << "Bad height";
    EXPECT_EQ(c.origin(), coord(0.f, 0.f, 0.f)) << "Bad origin";
    EXPECT_EQ(c.steering(), coord(0.f, 0.f, 1.f)) << "Bad steering";
    }

    {
    camera c(640, 480, {1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f});
    EXPECT_EQ(c.width(), 640) << "Bad width";
    EXPECT_EQ(c.height(), 480) << "Bad height";
    EXPECT_EQ(c.origin(), coord(1.f, 1.f, 1.f)) << "Bad origin";
    EXPECT_EQ(c.steering(), coord(-1.f, -1.f, -1.f)) << "Bad steering";
    }
}

TEST(camera, ray_generation)
{
    camera c(640, 480);


}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
