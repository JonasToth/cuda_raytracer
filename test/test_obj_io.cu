#include "gtest/gtest.h"

#include "obj_io.h"
#include <algorithm>
#include <vector>

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

TEST(obj_io, loading_simple) {
    world_geometry w;
    w.load("shapes.obj");

    EXPECT_EQ(w.vertex_count(), 122) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 228) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 4) << "Bad number of Shapes";

    const thrust::host_vector<coord> h_vertices     = w.vertices();
    const std::vector<coord> vertices(h_vertices.begin(), h_vertices.end());



    //const thrust::host_vector<triangle> h_triangles = w.triangles();
    //const std::vector<triangle> triangles(h_triangles.begin(), h_triangles.end());

    //std::clog << "Back conversion to host memory done" << std::endl;

    //const bool isValidRange = std::all_of(triangles.begin(), triangles.end(), 
                                          //[](const triangle& t) { return t.isValid(); });
    //EXPECT_EQ(isValidRange, true) << "Invalid triangles found";
}

TEST(obj_io, test_bad_input) {
    world_geometry w;

    w.load("bad.obj");
    //ASSERT_THROW(w.load("bad.obj"), std::invalid_argument) << "Did not notice the quad";
}

TEST(obj_io, loading_complex) {
    world_geometry w;
    w.load("mini_cooper.obj");

    EXPECT_EQ(w.vertex_count(), 234435) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(),304135) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 49) << "Bad number of Shapes";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
