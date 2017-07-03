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

    auto r = c.rayAt(320, 240);
    ASSERT_EQ(r.origin, coord(0.f, 0.f, 0.f)) << "Not from center of the universe";
    ASSERT_EQ(r.direction, coord(0.f, 0.f, 1.f)) << "Centered ray not in optical axis";

    std::clog << c.rayAt(0, 0).direction << std::endl;
    std::clog << c.rayAt(640, 0).direction << std::endl;
    std::clog << c.rayAt(0, 480).direction << std::endl;
    std::clog << c.rayAt(640, 480).direction << std::endl;
}

TEST(camera, complex_rays)
{
    // look at center of the world
    camera c(640, 480, {10.f, 10.f, 10.f}, {-10.f, -10.f, -10.f});

    auto r = c.rayAt(320, 240);
    ASSERT_EQ(r.origin, coord(10.f, 10.f, 10.f)) << "Not from center of camera";
    ASSERT_EQ(r.direction, normalize(coord(-1.f, -1.f, -1.f))) 
              << "Centered ray not in optical axis";

    std::clog << c.rayAt(0, 0).direction << std::endl;
    std::clog << c.rayAt(640, 0).direction << std::endl;
    std::clog << c.rayAt(0, 480).direction << std::endl;
    std::clog << c.rayAt(640, 480).direction << std::endl;
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
