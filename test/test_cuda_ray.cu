#include "gtest/gtest.h"
#include "ray.h"

TEST(cuda_ray, init)
{
    ray r;
    r.origin = coord{0, 0, -15};
    r.direction = coord{0, 0, 1};
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
