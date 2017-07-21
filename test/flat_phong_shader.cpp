#include "graphic/render/shading.h"
#include "management/memory_surface.h"
#include "management/world.h"
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    const std::size_t width = 800, height = 600;
    const std::string obj_name(argv[1]);
    const std::string img_name(argv[2]);

    memory_surface render_surface(width, height);
    world_geometry scene(obj_name);

    // Light Setup similar to blender (position and stuff taken from there)
    const coord camera_posi = {-1.5f, 1.2f, -1.5f};
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), coord(-1.1f,  1.1,  1.1f));
    scene.add_light(phong_light(spec, diff), coord( 1.1f, -1.1, -1.1f));
    scene.add_light(phong_light(spec, diff), coord(-1.1f, -1.1,  1.1f));
    scene.add_light(phong_light(spec, diff), coord(-1.1f, -1.1, -1.1f));
    scene.set_camera(camera(width, height, camera_posi, coord(0.f, 0.f, 0.f) - camera_posi));

    render_flat(render_surface, scene.handle());
    render_surface.save_as_png(img_name);

    return 0;
}
