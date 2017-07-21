#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <gsl/gsl>

#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "macros.h"
#include <algorithm>

/** @file test/test_build_cuda.cpp
 * Test if cuda is found on the system and is useable on the machine.
 */

TEST(CUDA, init)
{
    int NbrDevices;
    cudaGetDeviceCount(&NbrDevices);
    ASSERT_GT(NbrDevices, 0) << "No Cuda devices were found";
}

// copied from
TEST(CUDA, thrust_call)
{
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

struct ASimpleClass {
    __host__ __device__ ASimpleClass(int value)
      : value{value}
    {
    }
    int value;
};

struct SimpleClass {
    __host__ __device__ SimpleClass(ASimpleClass* common_constant)
      : value{0}
      , common_constant{common_constant}
    {
    }

    __host__ __device__ SimpleClass(int v, ASimpleClass* common_constant)
      : value{v}
      , common_constant{common_constant}
    {
    }

    __host__ __device__ SimpleClass(const SimpleClass& o)
      : value{o.value}
      , common_constant{o.common_constant}
    {
    }

    int value;
    ASimpleClass* common_constant;
};

struct Square {
    CUCALL SimpleClass operator()(const SimpleClass& o)
    {
        return SimpleClass{o.value * o.value + o.common_constant->value, o.common_constant};
    }
};

TEST(CUDA, thrust_with_object)
{
    const auto cvptr = thrust::device_malloc<ASimpleClass>(1);
    const auto csptr = thrust::device_new(cvptr, ASimpleClass{42});

    const auto vptr = thrust::device_malloc<SimpleClass>(100);
    const auto sptr = thrust::device_new(vptr, SimpleClass{csptr.get()}, 100);

    thrust::fill(sptr, sptr + 100, SimpleClass{15, csptr.get()});
    thrust::transform(sptr, sptr + 100, sptr, Square{});

    thrust::device_free(cvptr);
    thrust::device_free(vptr);
}

TEST(CUDA, thrust_with_vector)
{
    const auto cvptr = thrust::device_malloc<ASimpleClass>(1);
    const auto csptr = thrust::device_new(cvptr, ASimpleClass{42});

    thrust::device_vector<SimpleClass> vec(100, SimpleClass{15, csptr.get()});

    thrust::fill(vec.begin(), vec.end(), SimpleClass{30, csptr.get()});
    thrust::transform(vec.begin(), vec.end(), vec.begin(), Square{});

    thrust::device_free(cvptr);
}
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
