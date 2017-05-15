#include "gtest/gtest.h"
#include <gsl/gsl>
#include <GLFW/glfw3.h>

TEST(GLFW, init) {
    auto init_val = glfwInit();
    ASSERT_NE(init_val, 0) << "Could not initialize GLFW";
    glfwTerminate();
}

TEST(GLFW, window) {
    auto init_val = glfwInit();
    ASSERT_NE(init_val, 0) << "Could not initialize GLFW";

    gsl::owner<GLFWwindow*> window = glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
    ASSERT_NE(window, nullptr) << "Window not created";
    
    // cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
