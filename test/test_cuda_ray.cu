#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <utility>

#include "ray.h"

constexpr std::size_t SquareDim = 10;

struct does_intersect {
    CUCALL bool operator()(const LIB::pair<bool, intersect>& r) { return r.first; };
};
struct has_good_depth {
    CUCALL bool operator()(const LIB::pair<bool, intersect>& r) 
    { return !r.first || (r.second.depth >= 1.f); };
};

struct fire_ray {
    CUCALL fire_ray(const thrust::device_ptr<triangle> T) : T{T} {}
    CUCALL ~fire_ray() = default;

    CUCALL LIB::pair<bool, intersect> operator()(const ray& Ray) 
    { return Ray.intersects(*T); }

    const thrust::device_ptr<triangle> T;
};

TEST(ray, init)
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

    bool DoesIntersect;
    intersect I;
    LIB::tie(DoesIntersect, I) = R.intersects(T);

    ASSERT_EQ(DoesIntersect, true) << "depth=" << I.depth;
    ASSERT_EQ(I.depth, 10.) << "(" << I.hit.x << "," << I.hit.y << "," << I.hit.z << ")\n" 
                            << "(" << I.normal.x << "," << I.normal.y << "," << I.normal.z << ")";
}

thrust::device_vector<ray> generateRays(const coord& Origin, std::size_t SquareDim) {
    // create multiple rays from the origin, 10x10 grid
    const float DY = 2.f / (SquareDim - 1);
    const float DX = 2.f / (SquareDim - 1);

    thrust::device_vector<ray> AllRays(SquareDim * SquareDim);
    std::size_t Index = 0;
    for(float Y = 1.f; Y > -1.f; Y-= DY)
    {
        for(float X = -1.f; X < 1.f; X+= DX)
        {
            const coord Dir{X, Y, 1.f};
            AllRays[Index] = ray{Origin, Dir};
            ++Index;
        }
    }
    return AllRays;
}

thrust::device_vector<LIB::pair<bool, intersect>> 
traceTriangle(const thrust::device_ptr<triangle> T, const thrust::device_vector<ray>& AllRays)
{
    OUT << "Before trace" << std::endl;
    // raytrace all the rays, and save result
    thrust::device_vector<LIB::pair<bool, intersect>> Result(AllRays.size());
    OUT << "Space for result allocated" << std::endl;

    LIB::transform(AllRays.begin(), AllRays.end(), 
                   Result.begin(), fire_ray{T});

    OUT << "Done tracing" << std::endl;
    return Result;
}

std::string bwOutput(const thrust::device_vector<LIB::pair<bool, intersect>>& Result, 
                     std::size_t SquareDim)
{
#if 0
    std::vector<std::pair<bool, intersect>> HostResult(Result.size());
    thrust::transform(Result.begin(), Result.end(), HostResult.begin(),
                      [] (const thrust::pair<bool, intersect>& R) {
                          return std::make_pair(R.first, R.second);
                      });
    OUT << "Data copied back" << std::endl;

    // output the data as "black white"
    std::size_t Index = 0;
    std::stringstream SS;
    for(std::size_t i = 0; i < SquareDim; ++i)
    {
        for(std::size_t j = 0; j < SquareDim; ++j)
        {
            bool DidHit;
            intersect I;
            std::tie(DidHit, I) = HostResult[Index];
            SS << (DidHit ? "*" : ".");
            ++Index;
        }
        SS << "\n";
    }

    OUT << "Done" << std::endl;
    return SS.str();
#else
    return "";
#endif
}

TEST(ray, trace_many_successfull)
{
    thrust::device_vector<coord> Vertices(4);
    Vertices[0] = {0,-1,1}; 
    Vertices[1] = {-1,1,1};
    Vertices[2] = {1,1,1};
    Vertices[3] = {0,0,2};
    const auto& P0 = Vertices[0];
    const auto& P1 = Vertices[1];
    const auto& P2 = Vertices[2];
    const auto& Origin = Vertices[3];

    triangle T{P0, P1, P2};
    const auto triangle_void = thrust::device_malloc(sizeof(triangle));
    const auto triangle_ptr = thrust::device_new(triangle_void, T);

    OUT << "Triangle and tracer origin created" << std::endl;

    const auto AllRays = generateRays(Origin, SquareDim);
    ASSERT_EQ(AllRays.size(), SquareDim * SquareDim);
    OUT << "Rays generated" << std::endl;
    const auto Result = traceTriangle(triangle_ptr, AllRays);
    ASSERT_EQ(Result.size(), SquareDim * SquareDim);
    OUT << "Raytracing done" << std::endl;

    const auto ContainsHit = LIB::any_of(thrust::device, Result.begin(), Result.end(), 
                                         does_intersect{});
    ASSERT_EQ(ContainsHit, true) << bwOutput(Result, SquareDim);

    const auto GoodDepth = LIB::all_of(thrust::device, Result.begin(), Result.end(),  
                                       has_good_depth{});
    ASSERT_EQ(GoodDepth, true) << bwOutput(Result, SquareDim);

    const auto HitCount = LIB::count_if(thrust::device, Result.begin(), Result.end(), 
                                        does_intersect{});
    ASSERT_GT(HitCount, 0.3 * SquareDim * SquareDim) << bwOutput(Result, SquareDim) +
                                                        "More hits are expected\n";
    ASSERT_LT(HitCount, 0.8 * SquareDim * SquareDim) << bwOutput(Result, SquareDim) +
                                                        "Less hits are expected\n";


    //std::cout << bwOutput(Result, SquareDim) << std::endl;
    OUT << "BW output done" << std::endl;
}

/*
TEST(ray, trace_many_failing)
{
    thrust::device_vector<coord> Vertices(4);
    Vertices[0] = {0,-1,1}; 
    Vertices[1] = {-1,1,1};
    Vertices[2] = {1,1,1};
    Vertices[3] = {0,0,2};
    const auto& P0 = Vertices[0];
    const auto& P1 = Vertices[1];
    const auto& P2 = Vertices[2];
    const auto& Origin = Vertices[3];

    triangle T{P0, P1, P2};

    const auto AllRays = generateRays(Origin, SquareDim);
    const auto Result = traceTriangle(T, AllRays);
    
    //std::cout << bwOutput(Result, SquareDim) << std::endl;
    const auto ContainsHit = LIB::any_of(Result.begin(), Result.end(), does_intersect{});
    ASSERT_EQ(ContainsHit, false) << bwOutput(Result, SquareDim);
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
