#include "graphic/kernels/shaded.h"
#include "graphic/kernels/trace.h"
#include "graphic/kernels/utility.h"
#include "graphic/render/shading.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "management/world.h"
#include "scene_setup.h"
#include "util/demos/fps_demo.h"

#include <chrono>
#include <string>
#include <thread>

std::string get_scene_name(int argc, char** argv)
{
    if (argc == 1)
        return std::string("materials_flat.obj");
    else
        return std::string(argv[1]);
}

int main(int argc, char** argv)
{
    bool camera_changed = true;
    window win(800, 600, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);
    glfwSetKeyCallback(w, register_key_press);

    // Camera Setup similar to blender
    camera c(win.getWidth(), win.getHeight());
    surface_raii render_surface(win.getWidth(), win.getHeight());

    std::clog << "Setup Rendering Platform initialized" << std::endl;

    world_geometry scene(get_scene_name(argc, argv));
    setup_common_scene(scene);

    std::clog << "World initialized" << std::endl;

    auto render_lambda = [&]() {
        render_smooth<hard_shadow_tag>(render_surface.getSurface(), scene.handle());

        render_surface.render_gl_texture();
        glfwSwapBuffers(w);

        std::clog << "World rendered" << std::endl;
        std::clog << "Camera Position: " << c.origin() << std::endl;
        std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;

        camera_changed = false;
    };

    render_lambda();
    while (!glfwWindowShouldClose(w)) {
        camera_changed = handle_keys(w, c);
        if (camera_changed) {
            scene.set_camera(c);
            render_lambda();
        }

        glfwWaitEvents();
    }
    input_manager::instance().clear();

    return 0;
}
