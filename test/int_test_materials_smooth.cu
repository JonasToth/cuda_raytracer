#include "util/tests/integration_render.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Provide test name, otherwise file io can not succeed!" << std::endl;
        return 1;
    }

    integration_render materials_smooth(argv[1]);

    materials_smooth.init_default();
    materials_smooth.run();

    return 0;
} 
