#include "gtest/gtest.h"

#include "graphic/camera.h"
#include "management/input_manager.h"

TEST(camera_control, move)
{
    camera c(640, 480);

    c.move({1.f, 0.f, 0.f});
    EXPECT_EQ(c.origin(), coord(1.f, 0., 0.f)) << "Bad movement";
}

TEST(camera_control, turn)
{

}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
