#include "graphic/render/shading.h"
#include "management/memory_surface.h"
#include "management/world.h"
#include "scene_setup.h"
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    const std::size_t width = 2000, height = 1000;
    const std::string obj_name(argv[1]);
    const std::string img_name(argv[2]);

    memory_surface render_surface(width, height);

    world_geometry scene(obj_name);
    setup_common_scene(scene);

    // Light Setup similar to blender (position and stuff taken from there)
    equirectengular c(width, height, coord(0.0f, 0.0f, 18.0f), coord(0.0f, 0.0f, 1.0f));

    render_smooth<hard_shadow_tag>(render_surface, c, scene.handle());
    render_surface.save_as_png(img_name);

    return 0;
}
