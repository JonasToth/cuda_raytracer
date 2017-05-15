#include "gtest/gtest.h"
#include <GLFW/glfw3.h>
#include <gsl/gsl>

TEST(GLFW, init) {
    auto InitVal = glfwInit();
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";
    glfwTerminate();
}

TEST(GLFW, window) {
    auto InitVal = glfwInit();
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    gsl::owner<GLFWwindow*> Window = glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
    ASSERT_NE(Window, nullptr) << "Window not created";
    
    // cleanup
    glfwDestroyWindow(Window);
    glfwTerminate();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
