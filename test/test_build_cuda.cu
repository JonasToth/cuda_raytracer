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

TEST(CUDA, thrust_call) {
    // generate 32M random numbers serially
    thrust::host_vector<int> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    ASSERT_TRUE(std::is_sorted(h_vec.begin(), h_vec.end())) << "Vector is not sorted";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
