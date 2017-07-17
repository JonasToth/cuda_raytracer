#include "util/tests/integration_render.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    integration_render cube_smooth(argv[1], argv[2]);

    cube_smooth.init_default();

    const coord position(-1.5f, -1.5f, -1.5f);
    auto& s = cube_smooth.getScene();

    s.set_camera(camera(800, 600, position, coord(1.f, 1.f, 1.f)));
    auto& l = s.lights();
    l.resize(1);

    float diff[3] = {0.9f, 0.9f, 0.9f};
    float spec[3] = {0.0f, 0.0f, 0.0f};
    l[0] = light_source(phong_light(spec, diff), position);

    cube_smooth.run();

    return 0;
}
