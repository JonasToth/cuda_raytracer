/** Main Source for executable. */


#include <iostream>

auto feature() { return 42; }

int main(int  /*argc*/, char**  /*argv*/) 
{
    std::cout << "Hello world, lets raytrace" << std::endl;
    std::cout << feature() << std::endl;
    return 0;
}
