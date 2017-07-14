#include "gtest/gtest.h"

#include "graphic/camera.h"
#include "management/input_manager.h"

TEST(camera_control, move)
{
    camera c(640, 480);

    c.move({1.f, 0.f, 0.f});
    EXPECT_EQ(c.origin(), coord(1.f, 0., 0.f)) << "Bad movement";
}

TEST(camera_control, yawing)
{
    const float deg45 = M_PI / 4.f;
    const float deg90 = M_PI / 2.f;

    camera c(640, 480);
    const int cx = 320, cy = 240;
    const coord original_steering = c.steering();


    c.turn(deg45, 0.f);
    EXPECT_EQ(c.steering(), normalize({1.f, 0.f, 1.f}))
        << "Yawing does not the expected thing";
    c.turn(-deg45, 0.f);
    EXPECT_EQ(c.steering(), normalize({0.f, 0.f, 1.f}))
        << "Camera not looking in z direction";
    EXPECT_EQ(c.steering(), original_steering) << "Back rotation does not restore";

    c.turn(deg90, 0.f);
    EXPECT_EQ(c.steering(), coord(1.f, 0.f, 0.f)) << "Turning yaw doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(1.f, 0.f, 0.f))
        << "Turning does not affect generated ray correctly";

    c.turn(-deg90, 0.f);
    EXPECT_EQ(c.steering(), coord(0.f, 0.f, 1.f)) << "Turning yaw doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(0.f, 0.f, 1.f))
        << "Turning does not affect generated ray correctly";

    c.turn(-deg90, 0.f);
    EXPECT_EQ(c.steering(), coord(-1.f, 0.f, 0.f)) << "Turning yaw doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(-1.f, 0.f, 0.f))
        << "Turning does not affect generated ray correctly";
}

TEST(camera_control, pitching)
{
    const float deg45 = M_PI / 4.f;
    const float deg90 = M_PI / 2.f;

    camera c(640, 480);
    const int cx = 320, cy = 240;
    const coord original_steering = c.steering();

    c.turn(0.f, deg45);
    EXPECT_EQ(c.steering(), normalize({0.f, -1.f, 1.f}))
        << "Pitching does not the expected thing";
    c.turn(0.f, -deg45);
    EXPECT_EQ(c.steering(), normalize({0.f, 0.f, 1.f}))
        << "Camera not looking in z direction";
    EXPECT_EQ(c.steering(), original_steering) << "Back rotation does not restore";

    c.turn(0.f, deg90);
    EXPECT_EQ(c.steering(), coord(0.f, -1.f, 0.f)) << "Turning pitch doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(0.f, -1.f, 0.f))
        << "Turning does not affect generated ray correctly";

    c.turn(0.f, -deg90);
    EXPECT_EQ(c.steering(), coord(0.f, 0.f, 1.f)) << "Turning pitch doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(0.f, 0.f, 1.f))
        << "Turning does not affect generated ray correctly";

    c.turn(0.f, -deg90);
    EXPECT_EQ(c.steering(), coord(0.f, 1.f, 0.f)) << "Turning pitch doesnt work";
    EXPECT_EQ(c.rayAt(cx, cy).direction, coord(0.f, 1.f, 0.f))
        << "Turning does not affect generated ray correctly";
}

TEST(camera_control, yaw_and_pitch)
{
    const float deg45 = M_PI / 4.f;
    camera c(640, 480);
    // const int cx = 320, cy = 240;
    // const coord original_steering = c.steering();

    c.turn(deg45, deg45);
    EXPECT_DOUBLE_EQ(norm(c.steering()), 1.f) << "No unit vector for steering";
    // EXPECT_EQ(c.steering(), normalize({1.f, 1.f, 1.f}))
    //<< "Yawing and Pitching does not the expected thing";
    // EXPECT_EQ(c.rayAt(cx, cy).direction, coord(1.f, -1.f, 1.f))
    //<< "Turning does not affect generated ray correctly";

    c.turn(-deg45, -deg45);
    EXPECT_DOUBLE_EQ(norm(c.steering()), 1.f) << "No unit vector for steering";
    // EXPECT_EQ(c.steering(), normalize({0.f, 0.f, 1.f}))
    //<< "Camera not looking in z direction";
    // EXPECT_EQ(c.rayAt(cx, cy).direction, coord(0.f, 0.f, 1.f))
    //<< "Turning does not affect generated ray correctly";
    // EXPECT_EQ(c.steering(), original_steering) << "Back rotation does not restore";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
