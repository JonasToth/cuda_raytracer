#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "management/surface_raii.h"


TEST(surface, basic_properties)
{
    ASSERT_THROW(surface_raii vis(640, 480), std::runtime_error) 
                 << "No OpenGL context must be found";
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
