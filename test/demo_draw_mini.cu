#include "graphic/kernels/trace.h"
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
    window win(640, 480, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);
    glfwSetKeyCallback(w, register_key_press);

    // Camera Setup similar to blender
    camera c(win.getWidth(), win.getHeight(), {-2.0f, -2.0f, -3.0f}, { 1.0f,  1.0f, -1.0f});
    surface_raii render_surface(win.getWidth(), win.getHeight());

    std::clog << "Setup Rendering Platform initialized" << std::endl;

    //world_geometry scene("mini_cooper.obj");
    world_geometry scene("mini_reduced.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), coord( 2.0f,  2.0f, -2.0f));
    scene.add_light(phong_light(spec, diff), coord(-2.0f,  2.0f, -2.0f));
    //scene.add_light(phong_light(spec, diff), coord(-1.3f, 0.8f, 2.0f));
    //scene.add_light(phong_light(spec, diff), coord(-1.7f, -1.7f, 0.8f));

    std::clog << "World initialized" << std::endl;

    auto render_lambda = [&]() {
        render_flat<no_shadow_tag>(render_surface, c, scene.handle());
        std::this_thread::sleep_for(std::chrono::seconds(10));

        render_surface.render_gl_texture();
        glfwSwapBuffers(w);
        //render_surface.save_as_png("mini.png");
        render_surface.save_as_png("mini_reduced.png");

        std::clog << "World rendered" << std::endl;
        std::clog << "Camera Position: " << c.origin() << std::endl;
        std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
        camera_changed = false;
    };


    render_lambda();
    while (!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
        camera_changed = handle_keys(w, c);
        if (camera_changed)
            render_lambda();
    }

    input_manager::instance().clear();
    return 0;
}
