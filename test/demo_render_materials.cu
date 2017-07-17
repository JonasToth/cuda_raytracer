#include "graphic/kernels/shaded.h"
#include "graphic/kernels/trace.h"
#include "graphic/kernels/utility.h"
#include "graphic/render/shading.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "management/world.h"
#include "util/demos/fps_demo.h"

#include <chrono>
#include <thread>


int main(int argc, char** argv)
{
    bool camera_changed = true;
    window win(800, 600, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);
    glfwSetKeyCallback(w, register_key_press);

    // Camera Setup similar to blender
    camera c(win.getWidth(), win.getHeight(), {0.0f, 0.5f, 2.5f}, {0.1f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());

    std::clog << "Setup Rendering Platform initialized" << std::endl;

    world_geometry scene("material_scene.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {-1.7f, -1.5f, -1.5f});
    scene.add_light(phong_light(spec, diff), {1.3f, -1.8f, -1.2f});
    scene.add_light(phong_light(spec, diff), {-1.1f, 2.0f, 1.1f});
    scene.add_light(phong_light(spec, diff), {-1.5f, -1.5f, 1.5f});

    std::clog << "World initialized" << std::endl;

    auto render_lambda = [&]() {
        const auto& triangles = scene.triangles();
        render_flat(render_surface.getSurface(), scene.handle());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        render_surface.render_gl_texture();
        glfwSwapBuffers(w);
        std::clog << "World rendered" << std::endl;
        std::clog << "Camera Position: " << c.origin() << std::endl;
        std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
        camera_changed = false;
    };


    while (!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
        camera_changed = handle_keys(w, c);
        if (camera_changed)
            render_lambda();
    }
    input_manager::instance().clear();

    return 0;
}
