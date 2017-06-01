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

TEST(ray, intersection)
{
    ray R;
    R.origin    = coord{0, 0, 0};
    R.direction = coord{0, 0, 1};

    const coord P0{0, -10, 10}, P1{-10, 10, 10}, P2{10, 10, 10};
    triangle T{P0, P1, P2};

    auto DoesIntersect = false;
    intersect I;
    LIB::tie(DoesIntersect, I) = R.intersects(T);

    ASSERT_EQ(DoesIntersect, true);
    ASSERT_EQ(I.depth, 10.);

    std::cout << I.depth << std::endl;
    std::cout << "(" << I.hit.x << "," << I.hit.y << "," << I.hit.z << ")" << std::endl;
    std::cout << "(" << I.normal.x << "," << I.normal.y << ","
              << I.normal.z << ")" << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
