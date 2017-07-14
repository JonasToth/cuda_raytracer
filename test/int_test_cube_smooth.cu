#include "util/tests/integration_render.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    integration_render cube_smooth(argv[1]);

    cube_smooth.init_default();
    cube_smooth.run();

    return 0;
}
