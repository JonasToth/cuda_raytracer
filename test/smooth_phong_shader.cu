#include "graphic/render/shading.h"
#include "management/surface_raii.h"
#include "management/window.h"
#include "management/world.h"
#include "scene_setup.h"
#include <chrono>
#include <string>
#include <thread>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    const std::size_t width = 800, height = 600;
    const std::string obj_name(argv[1]);
    const std::string img_name(argv[2]);

    window w(width, height, "Integration Test " + obj_name);
    glfwMakeContextCurrent(w.getWindow());

    surface_raii render_surface(width, height);

    world_geometry scene(obj_name);
    setup_common_scene(scene);

    // Light Setup similar to blender (position and stuff taken from there)
    scene.set_camera(camera(width, height));

    render_smooth<hard_shadow_tag>(render_surface, scene.handle());
    render_surface.render_gl_texture();
    glfwSwapBuffers(w.getWindow());

    render_surface.save_as_png(img_name);

    return 0;
}
