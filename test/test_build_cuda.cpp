#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** @file test/test_build_cuda.cpp
 * Test if cuda is found on the system and is useable on the machine.
 */

TEST(CUDA, init) {
    int NbrDevices;
    cudaGetDeviceCount(&NbrDevices);
    ASSERT_GT(NbrDevices, 0) << "No Cuda devices were found";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
