#include "gtest/gtest.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <utility>

#include "ray.h"

constexpr std::size_t SquareDim = 10;

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
traceTriangle(const triangle& T, const thrust::device_vector<ray>& AllRays)
{
    // raytrace all the rays, and save result
    thrust::device_vector<LIB::pair<bool, intersect>> Result(AllRays.size());

    LIB::transform(AllRays.begin(), AllRays.end(), 
                   Result.begin(), [T] CUCALL (const ray& Ray) {
                       return Ray.intersects(T);
                   });
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
    const coord P0{0, -1, 1}, P1{-1, 1, 1}, P2{1, 1, 1};
    triangle T{P0, P1, P2};
    
    const coord Origin{0, 0, 0};

    OUT << "Triangle and tracer origin created" << std::endl;

    const auto AllRays = generateRays(Origin, SquareDim);
    OUT << "Rays generated" << std::endl;
    const auto Result = traceTriangle(T, AllRays);
    OUT << "Raytracing" << std::endl;

    //std::cout << bwOutput(Result, SquareDim) << std::endl;
    OUT << "BW output done" << std::endl;
}

TEST(ray, trace_many_failing)
{
    const coord P0{0, -1, 1}, P1{-1, 1, 1}, P2{1, 1, 1};
    triangle T{P0, P1, P2};
    
    const coord Origin{0, 0, 2};

    const auto AllRays = generateRays(Origin, SquareDim);
    const auto Result = traceTriangle(T, AllRays);
    
    //std::cout << bwOutput(Result, SquareDim) << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
