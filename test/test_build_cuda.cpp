#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>


TEST(CUDA, init) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    ASSERT_GT(nDevices, 0) << "No Cuda devices were found";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
