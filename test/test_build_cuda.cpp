#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>


TEST(CUDA, init) {
    ASSERT_EQ(0,0) << "Basic test";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
