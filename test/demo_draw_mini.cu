#include "gtest/gtest.h"

#include "graphic/kernels/trace.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "management/world.h"
#include "util/kernel_launcher/world_shading.h"

#include <thread>
#include <chrono>

bool camera_changed = true;

static void handle_keys(GLFWwindow* w, camera& c)
{
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
}

int main(int argc, char** argv)
{
    window win(800, 600, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);
    glfwSetKeyCallback(w, register_key_press);

    // Camera Setup similar to blender
    camera c(win.getWidth(), win.getHeight(), 
             {-2.5f, 3.f, 3.f}, {0.0f, -0.1f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());

    std::clog << "Setup Rendering Platform initialized" << std::endl;
    
    world_geometry scene("mini_cooper.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), coord(0.8f, 0.9f, 1.5f));
    scene.add_light(phong_light(spec, diff), coord(1.7f, -1.1f, -0.3f));
    scene.add_light(phong_light(spec, diff), coord(-1.3f, 0.8f, 2.0f));
    scene.add_light(phong_light(spec, diff), coord(-1.7f, -1.7f, 0.8f));

    std::clog << "World initialized" << std::endl;

    auto render_lambda = [&]() {
        const auto& triangles = scene.triangles();
        raytrace_many_shaded(render_surface.getSurface(), c,
                             triangles.data().get(), triangles.size(),
                             scene.lights().data().get(), scene.light_count());
        render_surface.render_gl_texture();
        render_surface.save_as_png("mini.png");
        //glfwSwapBuffers(w);
        std::clog << "World rendered" << std::endl;
        std::clog << "Camera Position: " << c.origin() << std::endl;
        std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
        camera_changed = false;
    };


    render_lambda();
    while(!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
        handle_keys(w, c);
        if(camera_changed)
            render_lambda();
    } 
    
    input_manager::instance().clear();
    return 0;
}


