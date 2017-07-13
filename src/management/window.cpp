#include "window.h"
#include <stdexcept>

window::window(int width, int height, const std::string& title, bool visible)
    : __width{width}
    , __height{height}
{
    auto init = glfwInit();

    if(init == 0)
        throw std::runtime_error{"Could not initialize glfw"};

    // invisible windows for test cases and only image rendering
    if(!visible)
        glfwWindowHint(GLFW_VISIBLE, false);

    __w = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

    if(__w == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error{"Could not create a window"};
    }
}

window::~window() 
{
    glfwDestroyWindow(__w);
    glfwTerminate();
}
