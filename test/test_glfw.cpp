#include "gtest/gtest.h"
#include <GLFW/glfw3.h>
#include <gsl/gsl>

TEST(GLFW, init) {
    auto InitVal = glfwInit();
    auto _ = gsl::finally([](){ glfwTerminate(); });
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";
}

TEST(GLFW, window) {
    auto InitVal = glfwInit();
    auto A = gsl::finally([](){ glfwTerminate(); });
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    gsl::owner<GLFWwindow*> Window = glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
    auto B = gsl::finally([Window](){ glfwDestroyWindow(Window); });
    ASSERT_NE(Window, nullptr) << "Window not created";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
