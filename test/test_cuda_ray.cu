#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "ray.h"

TEST(cuda_ray, init)
{
    ray R;
    R.origin = coord{0, 0, -15};
    R.direction = coord{0, 0, 1};
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
