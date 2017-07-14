
inline bool handle_keys(GLFWwindow* w, camera& c)
{
    bool camera_changed = false;
    const auto& im = input_manager::instance();

    const float dP = 0.5;
    if(im.isPressed(GLFW_KEY_ESCAPE))
        glfwSetWindowShouldClose(w, GLFW_TRUE);

    else if(im.isPressed(GLFW_KEY_A))
    {
        c.move({-dP, 0.f, 0.f});
        camera_changed = true;
    }
    else if(im.isPressed(GLFW_KEY_D))
    {
        c.move({dP, 0.f, 0.f});
        camera_changed = true;
    }
    else if(im.isPressed(GLFW_KEY_W))
    {
        c.move({0.f, dP, 0.f});
        camera_changed = true;
    }
    else if(im.isPressed(GLFW_KEY_S))
    {
        c.move({0.f, -dP, 0.f});
        camera_changed = true;
    }
    else if(im.isPressed(GLFW_KEY_Q))
    {
        c.move({0.f, 0.f, dP});
        camera_changed = true;
    }
    else if(im.isPressed(GLFW_KEY_E))
    {
        c.move({0.f, 0.f, -dP});
        camera_changed = true;
    }
    return camera_changed;
}
