#include "gtest/gtest.h"

#include "graphic/kernels/utility.h"
#include "graphic/kernels/utility.h"
#include "graphic/ray.h"
#include "graphic/triangle.h"
#include "management/memory_surface.h"
#include "management/window.h"

TEST(test_kernel, draw_triangle)
{
    const std::size_t width = 800, height = 600;

    // Use only CPU and RAM for raytracing
    memory_surface vis(width, height);

    // Create the Triangle and Coordinates on the device
    std::vector<coord> Vertices(5);
    Vertices[0] = {0, -1, 1};
    Vertices[1] = {-1, 1, 1};
    Vertices[2] = {1, 1, 1};
    Vertices[3] = {1, -0.8, 1};
    Vertices[4] = {-1, 0.8, 1};

    const auto* P0 = &Vertices[0];
    const auto* P1 = &Vertices[1];
    const auto* P2 = &Vertices[2];
    const auto* P3 = &Vertices[3];
    const auto* P4 = &Vertices[4];

    std::vector<coord> Normals(3);
    Normals[0] = normalize(cross(Vertices[1] - Vertices[0], Vertices[2] - Vertices[1]));
    Normals[1] = normalize(cross(Vertices[1] - Vertices[0], Vertices[3] - Vertices[0]));
    Normals[2] = normalize(cross(Vertices[2] - Vertices[4], Vertices[2] - Vertices[0]));

    const auto* t0_n = &Normals[0];
    const auto* t1_n = &Normals[1];
    const auto* t2_n = &Normals[2];

    std::vector<triangle> Triangles(3);
    Triangles[0] = triangle(P0, P1, P2, t0_n);
    Triangles[1] = triangle(P0, P1, P3, t1_n);
    Triangles[2] = triangle(P4, P2, P0, t2_n);

    black_kernel(vis);
    stupid_colors(vis, 15.f);


    vis.save_as_png("test_cpp_kernel.png");
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
