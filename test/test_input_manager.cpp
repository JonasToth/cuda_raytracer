#include "gtest/gtest.h"
#include "management/input_manager.h"
#include <GLFW/glfw3.h>

TEST(input_manager, key_handling)
{
    auto& m = input_manager::instance();

    m.press(GLFW_KEY_SPACE);
    m.press(GLFW_KEY_W);
    ASSERT_TRUE(m.isPressed(GLFW_KEY_SPACE)) << "Keypress not registered";
    ASSERT_TRUE(m.isPressed(GLFW_KEY_W)) << "Keypress not registered";
    ASSERT_FALSE(m.isPressed(GLFW_KEY_S)) << "not existing keypress registered";

    m.release(GLFW_KEY_SPACE);
    ASSERT_FALSE(m.isPressed(GLFW_KEY_SPACE)) << "Key release not registered";

    m.clear();
    ASSERT_FALSE(m.isPressed(GLFW_KEY_W)) << "Key not released after clearing";
    ASSERT_FALSE(m.isPressed(GLFW_KEY_S)) << "Key not released after clearing";
    ASSERT_FALSE(m.isPressed(GLFW_KEY_SPACE)) << "Key not released after clearing";
}

TEST(input_manager, mouse_handling)
{
    auto& m = input_manager::instance();
    EXPECT_DOUBLE_EQ(m.mouse_x(), 0.f) << "Bad initial mouse position";
    EXPECT_DOUBLE_EQ(m.mouse_y(), 0.f) << "Bad initial mouse position";
    EXPECT_DOUBLE_EQ(m.mouse_diff_x(), 0.f) << "Bad initial mouse position";
    EXPECT_DOUBLE_EQ(m.mouse_diff_y(), 0.f) << "Bad initial mouse position";
    
    m.move_mouse(100., 200.);
    EXPECT_DOUBLE_EQ(m.mouse_x(), 100.) << "Movement not accounted";
    EXPECT_DOUBLE_EQ(m.mouse_y(), 200.) << "Movement not accounted";
    EXPECT_DOUBLE_EQ(m.mouse_diff_x(), 100.) << "Difference not calculated correctly";
    EXPECT_DOUBLE_EQ(m.mouse_diff_y(), 200.) << "Difference not calculated correctly";
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
