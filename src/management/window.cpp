#include "window.h"
#include <stdexcept>

window::window(int width, int height, const std::string& title)
  : __width{width}
  , __height{height}
{
    auto init = glfwInit();

    if (init == 0)
        throw std::runtime_error{"Could not initialize glfw"};

    // glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    __w = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

    if (__w == nullptr) {
        glfwTerminate();
        throw std::runtime_error{"Could not create a window"};
    }
    glfwMakeContextCurrent(__w);
}

window::~window()
{
    glfwDestroyWindow(__w);
    glfwTerminate();
}
