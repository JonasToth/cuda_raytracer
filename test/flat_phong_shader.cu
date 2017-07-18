#include "graphic/render/shading.h"
#include "management/surface_raii.h"
#include "management/window.h"
#include "management/world.h"
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

    // Light Setup similar to blender (position and stuff taken from there)
    const coord camera_posi = {-1.5f, 1.2f, -1.5f};
    float spec[3] = {0.4f, 0.4f, 0.4f};
    float diff[3] = {0.4f, 0.4f, 0.4f};
    scene.add_light(phong_light(spec, diff), camera_posi);
    scene.set_camera(camera(width, height, camera_posi, coord(0.f, 0.f, 0.f) - camera_posi));

    render_flat(render_surface.getSurface(), scene.handle());
    render_surface.render_gl_texture();
    std::this_thread::sleep_for(std::chrono::seconds(5));

    render_surface.save_as_png(img_name);

    return 0;
}