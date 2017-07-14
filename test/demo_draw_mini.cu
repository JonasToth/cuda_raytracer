#include "graphic/kernels/trace.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "management/world.h"
#include "util/demos/fps_demo.h"
#include "util/kernel_launcher/world_shading.h"

#include <thread>
#include <chrono>

int main(int argc, char** argv)
{
    bool camera_changed = true;
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
        raytrace_many_shaded(render_surface.getSurface(), scene.handle());
        render_surface.render_gl_texture();
        render_surface.save_as_png("mini.png");
        glfwSwapBuffers(w);
        std::clog << "World rendered" << std::endl;
        std::clog << "Camera Position: " << c.origin() << std::endl;
        std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
        camera_changed = false;
    };


    render_lambda();
    while(!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
        camera_changed = handle_keys(w, c);
        if(camera_changed)
            render_lambda();
    } 
    
    input_manager::instance().clear();
    return 0;
}


