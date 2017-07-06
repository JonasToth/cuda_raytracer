#include "gtest/gtest.h"
#include <GLFW/glfw3.h>
#include <gsl/gsl>

#include "management/window.h"

TEST(GLFW, init) {
    auto InitVal = glfwInit();
    auto _ = gsl::finally([](){ glfwTerminate(); });
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    glfwTerminate();
}

TEST(GLFW, window) {
    auto InitVal = glfwInit();
    auto A = gsl::finally([](){ glfwTerminate(); });
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    gsl::owner<GLFWwindow*> Window = glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
    auto B = gsl::finally([Window](){ glfwDestroyWindow(Window); });
    ASSERT_NE(Window, nullptr) << "Window not created";

    glfwTerminate();
}

TEST(GLFW, wrapper) {
    window w(640, 480, "Title");
    ASSERT_NE(w.getWindow(), nullptr) << "Could not create window";

    ASSERT_EQ(w.getWidth(), 640) << "Incorrect width";
    ASSERT_EQ(w.getHeight(), 480) << "Incorrect height";
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
