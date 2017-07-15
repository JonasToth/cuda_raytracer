#include "util/tests/integration_render.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    integration_render materials(argv[1]);

    materials.init_default();
    materials.run();

    return 0;
}
