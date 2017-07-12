#include "gtest/gtest.h"
#include "graphic/ray.h"

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
    const coord normal = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2};
    T.normal(&normal);

    bool DoesIntersect;
    intersect I;
    std::tie(DoesIntersect, I) = R.intersects(T);

    EXPECT_EQ(DoesIntersect, true) << "depth=" << I.depth;
    EXPECT_EQ(I.hit, coord(0.f, 0.f, 10.f)) << "Hit coordinates not correct " << I.hit;
    EXPECT_EQ(I.depth, 10.) << I.hit << "\n"
                            << I.normal;
    EXPECT_EQ(I.normal, coord(0., 0., -1.f)) << I.normal;

    R.direction = normalize(coord{0.f, 0.5f, 1.f});
    std::tie(DoesIntersect, I) = R.intersects(T);
    
    EXPECT_EQ(DoesIntersect, true) << "Intersection wrong";
    EXPECT_EQ(I.hit, coord(0.f, 5.f, 10.f)) << "Hit coordinates not correct " << I.hit;
    EXPECT_EQ(I.depth, std::sqrt(125.f)) << "Hit depth wrong";


    R.origin = coord{1.f, 0.0f, 1.f};
    std::tie(DoesIntersect, I) = R.intersects(T);

    EXPECT_EQ(DoesIntersect, true) << "Intersection wrong";
    EXPECT_EQ(I.hit, coord(1.f, 4.5f, 10.f)) << "Hit coordinates not correct " << I.hit;
    EXPECT_LT(I.depth - std::sqrt(9.f * 9.f + 4.5f * 4.5f), 0.0001) << "Hit depth wrong";
}

std::vector<ray> generateRays(const coord& Origin, std::size_t SquareDim) {
    // create multiple rays from the origin, 10x10 grid
    const float DY = 2.f / (SquareDim - 1);
    const float DX = 2.f / (SquareDim - 1);

    std::vector<ray> AllRays(SquareDim * SquareDim);
    std::size_t Index = 0;
    for(float Y = 1.f; Y > -1.f; Y-= DY) // NOLINT
    {
        for(float X = -1.f; X < 1.f; X+= DX) // NOLINT
        {
            const coord Dir{X, Y, 1.f};
            AllRays.at(Index) = ray{Origin, Dir};
            ++Index;
        }
    }
    return AllRays;
}

std::vector<LIB::pair<bool, intersect>> traceTriangle(const triangle& T, 
                                                      const std::vector<ray>& AllRays)
{
    // raytrace all the rays, and save result
    LIB::vector<LIB::pair<bool, intersect>> Result(AllRays.size());

    LIB::transform(LIB::begin(AllRays), LIB::end(AllRays), 
                   LIB::begin(Result), [&T](const ray& Ray) {
                       return Ray.intersects(T);
                   });
    return Result;
}

std::string bwOutput(const std::vector<LIB::pair<bool, intersect>>& Result, 
                     std::size_t SquareDim)
{
    // output the data as "black white"
    std::size_t Index = 0;
    std::stringstream SS;
    for(std::size_t I = 0; I < SquareDim; ++I)
    {
        for(std::size_t J = 0; J < SquareDim; ++J)
        {
            bool DidHit; intersect HitResult;
            LIB::tie(DidHit, HitResult) = Result.at(Index);
            SS << (DidHit ? "*" : ".");
            ++Index;
        }
        SS << "\n";
    }
    return SS.str();
}

TEST(ray, trace_many_successfull)
{
    const coord P0{0, -1, 1}, P1{-1, 1, 1}, P2{1, 1, 1};
    const coord normal = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2};
    T.normal(&normal);
    
    const coord Origin{0, 0, 0};

    const auto AllRays = generateRays(Origin, SquareDim);
    const auto Result = traceTriangle(T, AllRays);
    
    const auto ContainsHit = LIB::any_of(LIB::begin(Result), LIB::end(Result), 
                             [](const LIB::pair<bool, intersect>& r) { return r.first; });
    ASSERT_EQ(ContainsHit, true) << bwOutput(Result, SquareDim);

    const auto GoodDepth = LIB::all_of(LIB::begin(Result), LIB::end(Result), 
                           [](const LIB::pair<bool, intersect>& r) 
                           { return !r.first || (r.second.depth >= 1.f); });
    ASSERT_EQ(GoodDepth, true) << bwOutput(Result, SquareDim);

    const auto HitCount = LIB::count_if(LIB::begin(Result), LIB::end(Result), 
                          [](const LIB::pair<bool, intersect>& r) { return r.first; });
    ASSERT_GT(HitCount, 0.3 * SquareDim * SquareDim) << bwOutput(Result, SquareDim) +
                                                        "More hits are expected\n";
    ASSERT_LT(HitCount, 0.8 * SquareDim * SquareDim) << bwOutput(Result, SquareDim) +
                                                        "Less hits are expected\n";

    std::cout << bwOutput(Result, SquareDim) << std::endl;
}

TEST(ray, trace_many_failing)
{
    const coord P0{0, -1, 1}, P1{-1, 1, 1}, P2{1, 1, 1};
    const coord normal = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2};
    T.normal(&normal);
    
    const coord Origin{0, 0, 10};

    const auto AllRays = generateRays(Origin, SquareDim);
    const auto Result = traceTriangle(T, AllRays);
    
    const auto ContainsHit = LIB::any_of(LIB::begin(Result), LIB::end(Result), 
                             [](const LIB::pair<bool, intersect>& r) { return r.first; });
    ASSERT_EQ(ContainsHit, false) << bwOutput(Result, SquareDim);
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}
