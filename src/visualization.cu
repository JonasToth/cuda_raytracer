#include "visualization.h"
#include <stdexcept>


namespace {
void quit_with_q(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_Q && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
}

visualization::visualization(int width, int height)
    : __width(width)
    , __height(height)
{
    auto init_value = glfwInit();

    if(init_value != 0)
        throw std::runtime_error{"Could not initialize glfw"};

    __window = glfwCreateWindow(__width, __height, "Cuda Raytracer", nullptr, nullptr);

    if(__window == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error{"Could not create window"};
    }

    glfwSetKeyCallback(__window, quit_with_q);
    glfwMakeContextCurrent(__window);

    while(!glfwWindowShouldClose(__window)) {
        glfwPollEvents();
    }
}

visualization::~visualization() 
{
    glfwDestroyWindow(__window);
    glfwTerminate();
}
