#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"
#include <gsl/gsl>
#include "graphic/triangle.h"
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

struct incorrect_triangle {
    CUCALL bool operator()(const triangle& T) {
        auto result = true;

        result&= T.isValid();

        result&= T.p0().x == 0;
        result&= T.p0().y == 0;
        result&= T.p0().z == 0;

        result&= T.p1().x == 1;
        result&= T.p1().y == 0;
        result&= T.p1().z == 0;

        result&= T.p2().x == 0;
        result&= T.p2().y == 1;
        result&= T.p2().z == 0;

        const auto N = T.normal();
        result&= N.x == 0;
        result&= N.y == 0;
        result&= N.z == 1;

        return !result;
    }
};

TEST(triangle_test, construction)
{
    thrust::device_vector<coord> Vertices(3);
    Vertices[0] = {0,0,0}; 
    Vertices[1] = {1,0,0};
    Vertices[2] = {0,1,0};
    const auto P0 = Vertices[0];
    const auto P1 = Vertices[1];
    const auto P2 = Vertices[2];

    thrust::device_vector<triangle> Triangles(1, {P0, P1, P2});

    OUT << "Constructed" << std::endl;

    ASSERT_EQ(thrust::none_of(thrust::device, Triangles.begin(), Triangles.end(),
                              incorrect_triangle{}), 
              true) << "Problematic triangle exists";
}

TEST(triangle_test, validity)
{
    thrust::device_vector<coord> Vertices(3, coord{0,0,0});
    const auto P0 = Vertices[0];
    const auto P1 = Vertices[1];
    const auto P2 = Vertices[2];

    const auto triangle_void = thrust::device_malloc(sizeof(triangle));
    auto _ = gsl::finally([&triangle_void]() { thrust::device_free(triangle_void); });
    const auto T = thrust::device_new(triangle_void, triangle{P0, P1, P2});

    OUT << "Data constructed and on the device" << std::endl;

    ASSERT_EQ(thrust::none_of(thrust::device, T, T+1,
                              incorrect_triangle{}), 
              false) << "Problematic triangle not detected";
}

struct does_contain_correct_points {
    CUCALL bool operator()(const triangle& T) {
        auto result = true;
        //T.contains(P0), true) << "P0";
        //T.contains(P1), true) << "P1";
        //T.contains(P2), true) << "P2";

        result&= T.contains({0.5, 0.5, 0});
        result&= T.contains({0.5, 0.0, 0});

        result&= !T.contains({0.5, 0.5, 1});
        result&= !T.contains({0.5, -0.5, 0});
        result&= !T.contains({-0.5, -0.5, 0});
        result&= !T.contains({-0.5, 0.5, 0});

        result&= !T.contains({-1, 0, 0});
        result&= !T.contains({2, 0, 0});
        result&= !T.contains({0.5, -1, 0});
        result&= !T.contains({0.5, 2, 0});

        return result;
    }
};

TEST(triangle_test, contains_point)
{
    thrust::device_vector<coord> Vertices(3, coord{0,0,0});
    Vertices[0] = {0,0,0}; 
    Vertices[1] = {1,0,0};
    Vertices[2] = {0,1,0};
    const auto P0 = Vertices[0];
    const auto P1 = Vertices[1];
    const auto P2 = Vertices[2];

    const auto triangle_void = thrust::device_malloc(sizeof(triangle));
    auto _ = gsl::finally([&triangle_void]() { thrust::device_free(triangle_void); });

    const auto T = thrust::device_new(triangle_void, triangle{P0, P1, P2});

    ASSERT_EQ(thrust::none_of(thrust::device, T, T + 1,
                              does_contain_correct_points{}), 
              true) << "Triangles do not recognize containing points correctly";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
