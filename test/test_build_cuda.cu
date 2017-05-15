#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

/** @file test/test_build_cuda.cpp
 * Test if cuda is found on the system and is useable on the machine.
 */

TEST(CUDA, init) {
    int NbrDevices;
    cudaGetDeviceCount(&NbrDevices);
    ASSERT_GT(NbrDevices, 0) << "No Cuda devices were found";
}

// copied from 
TEST(CUDA, thrust_call) {
    constexpr std::size_t VectorSize = 10000000u;

    // generate many random numbers
    thrust::host_vector<int> HVec(VectorSize);
    ASSERT_EQ(HVec.size(), VectorSize) << "Host vector not created with correct size";
    std::generate(HVec.begin(), HVec.end(), rand);

    // transfer data to the device
    thrust::device_vector<int> DVec = HVec;
    ASSERT_EQ(DVec.size(), VectorSize) << "Device Vector not created with correct size";

    // sort data on the device 
    thrust::sort(DVec.begin(), DVec.end());
    ASSERT_TRUE(thrust::is_sorted(DVec.begin(), DVec.end())) << "Sorted on GPU";

    // transfer data back to host
    thrust::copy(DVec.begin(), DVec.end(), HVec.begin());
    ASSERT_TRUE(std::is_sorted(HVec.begin(), HVec.end())) << "Vector is not sorted";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
