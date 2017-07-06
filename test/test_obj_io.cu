#include "gtest/gtest.h"

#include "material.h"
#include "obj_io.h"
#include <algorithm>
#include <vector>

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

TEST(obj_io, detail_load)
{
    thrust::host_vector<coord> h_vertices;
    thrust::host_vector<triangle> h_triangles;
    thrust::host_vector<phong_material> h_materials;
    std::size_t shape_count;

    __detail::deserialize_geometry("cube.obj", h_vertices, h_triangles, h_materials, shape_count);

    EXPECT_EQ(shape_count, 1) << "bad number of shapes";

    EXPECT_EQ(h_vertices[0], coord(1.f, -1.f, -1.f))            << "Bad vertex";
    EXPECT_EQ(h_vertices[1], coord(1.f, -1.f, 1.f))             << "Bad vertex";
    EXPECT_EQ(h_vertices[2], coord(-1.f, -1.f, 1.f))            << "Bad vertex";
    EXPECT_EQ(h_vertices[3], coord(-1.f, -1.f, -1.f))           << "Bad vertex";
    EXPECT_EQ(h_vertices[4], coord(1.f, 1.f, -0.999999f))       << "Bad vertex";
    EXPECT_EQ(h_vertices[5], coord(0.9999999f, 1.f, 1.000001f)) << "Bad vertex";
    EXPECT_EQ(h_vertices[6], coord(-1.f, 1.f, 1.f))             << "Bad vertex";
    EXPECT_EQ(h_vertices[7], coord(-1.f, 1.f, -1.f))            << "Bad vertex";

    EXPECT_EQ(h_triangles[0].p0(), h_vertices[1])   << "Bad Triangle v 0";
    EXPECT_EQ(h_triangles[0].p1(), h_vertices[3])   << "Bad Triangle v 0";
    EXPECT_EQ(h_triangles[0].p2(), h_vertices[0])   << "Bad Triangle v 0";

    std::clog << h_triangles[0].normal() << std::endl;

    EXPECT_EQ(h_triangles[1].p0(), h_vertices[7])   << "Bad Triangle v 1";
    EXPECT_EQ(h_triangles[1].p1(), h_vertices[5])   << "Bad Triangle v 1";
    EXPECT_EQ(h_triangles[1].p2(), h_vertices[4])   << "Bad Triangle v 1";

    std::clog << h_triangles[1].normal() << std::endl;

    EXPECT_EQ(h_triangles[2].p0(), h_vertices[4])   << "Bad Triangle v 2";
    EXPECT_EQ(h_triangles[2].p1(), h_vertices[1])   << "Bad Triangle v 2";
    EXPECT_EQ(h_triangles[2].p2(), h_vertices[0])   << "Bad Triangle v 2";

    std::clog << h_triangles[2].normal() << std::endl;
}

TEST(obj_io, load_cube) {
    world_geometry w("cube.obj");
    
    EXPECT_EQ(w.vertex_count(), 8) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 12) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 1) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 0) << "Bad number of materials";
}

TEST(obj_io, loading_simple) {
    world_geometry w;
    w.load("shapes.obj");

    EXPECT_EQ(w.vertex_count(), 122) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 228) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 4) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 0) << "Bad number of materials";

    const thrust::host_vector<coord> h_vertices     = w.vertices();
    const std::vector<coord> vertices(h_vertices.begin(), h_vertices.end());

    const thrust::host_vector<triangle> h_triangles = w.triangles();
    const std::vector<triangle> triangles(h_triangles.begin(), h_triangles.end());

    std::clog << "Back conversion to host memory done" << std::endl;

    const bool isValidRange = std::all_of(triangles.begin(), triangles.end(), 
                                          [](const triangle& t) { return t.isValid(); });
    EXPECT_EQ(isValidRange, true) << "Invalid triangles found";
}

TEST(obj_io, test_bad_input) {
    world_geometry w;

    w.load("bad.obj");
    //ASSERT_THROW(w.load("bad.obj"), std::invalid_argument) << "Did not notice the quad";
}

TEST(obj_io, test_simple_materials) {
    world_geometry w("test_camera_light.obj");

    EXPECT_EQ(w.vertex_count(), 8) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 12) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 1) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 1) << "Bad number of materials";
}

TEST(obj_io, loading_complex) {
    world_geometry w;
    w.load("mini_cooper.obj");

    EXPECT_EQ(w.vertex_count(), 234435) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(),304135) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 49) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 15) << "Bad number of materials";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
