#include "gtest/gtest.h"
#include "management/input_manager.h"
#include <GLFW/glfw3.h>

TEST(input_manager, basic_functionality)
{
    auto& m = input_manager::instance();

    m.press(GLFW_KEY_SPACE);
    m.press(GLFW_KEY_W);

    ASSERT_TRUE(m.isPressed(GLFW_KEY_SPACE)) << "Keypress not registered";
    ASSERT_TRUE(m.isPressed(GLFW_KEY_W)) << "Keypress not registered";
    ASSERT_FALSE(m.isPressed(GLFW_KEY_S)) << "not existing keypress registered";

    m.release(GLFW_KEY_SPACE);

    ASSERT_FALSE(m.isPressed(GLFW_KEY_SPACE)) << "Key release not registered";
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
