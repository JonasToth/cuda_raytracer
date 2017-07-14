#include "management/input_callback.h"
#include "management/input_manager.h"
#include <GLFW/glfw3.h>

void register_key_press(GLFWwindow* /*w*/, int key, int /*scancode*/, int action,
                        int /*mods*/)
{
    auto& im = input_manager::instance();

    if (action == GLFW_PRESS)
        im.press(key);
    else if (action == GLFW_RELEASE)
        im.release(key);
}


void register_mouse_movement(GLFWwindow* /*w*/, double xpos, double ypos)
{
    auto& im = input_manager::instance();
    im.move_mouse(xpos, ypos);
}
