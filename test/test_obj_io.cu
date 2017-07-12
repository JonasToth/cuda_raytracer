#include "gtest/gtest.h"

#include "graphic/material.h"
#include "obj_io.h"
#include <algorithm>
#include <vector>

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

TEST(obj_io, load_cube) {
    world_geometry w("cube.obj");
    
    EXPECT_EQ(w.vertex_count(), 8) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 12) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 1) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 0) << "Bad number of materials";

    const thrust::host_vector<coord> h_vertices     = w.vertices();
    const std::vector<coord> vertices(h_vertices.begin(), h_vertices.end());

    EXPECT_EQ(vertices[0], coord(1.f,       -1.f, -1.f));
    EXPECT_EQ(vertices[1], coord(1.f,       -1.f,  1.f));
    EXPECT_EQ(vertices[2], coord(-1.f,      -1.f,  1.f));
    EXPECT_EQ(vertices[3], coord(-1.f,      -1.f, -1.f));
    EXPECT_EQ(vertices[4], coord(1.f,        1.f, -0.9999999f));
    EXPECT_EQ(vertices[5], coord(0.999999f,  1.f,  1.000001f));
    EXPECT_EQ(vertices[6], coord(-1.f,       1.f,  1.f));
    EXPECT_EQ(vertices[7], coord(-1.f,       1.f, -1.f));


    const thrust::host_vector<triangle> h_triangles = w.triangles();
    const std::vector<triangle> triangles(h_triangles.begin(), h_triangles.end());
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

    // Test material properties
    const auto& d_material                          = w.materials();
    thrust::host_vector<phong_material> h_material  = w.materials();
    const thrust::host_vector<triangle> h_triangles = w.triangles();

    ASSERT_EQ(h_material.size(), 1) << "Inconsistent material counts!";

    // Material properties
    EXPECT_FLOAT_EQ(h_material[0].shininess(), 96.078431f);

    EXPECT_FLOAT_EQ(h_material[0].r.specular_reflection(), .5f);
    EXPECT_FLOAT_EQ(h_material[0].r.diffuse_reflection(), .64f);
    EXPECT_FLOAT_EQ(h_material[0].r.ambient_reflection(), 1.f);

    EXPECT_FLOAT_EQ(h_material[0].g.specular_reflection(), .5f);
    EXPECT_FLOAT_EQ(h_material[0].g.diffuse_reflection(), .64f);
    EXPECT_FLOAT_EQ(h_material[0].g.ambient_reflection(), 1.f);

    EXPECT_FLOAT_EQ(h_material[0].b.specular_reflection(), .5f);
    EXPECT_FLOAT_EQ(h_material[0].b.diffuse_reflection(), .64f);
    EXPECT_FLOAT_EQ(h_material[0].b.ambient_reflection(), 1.f);

    // Connection between triangle and material
    const auto& t = h_triangles[0];
    const auto m_ptr = &d_material[0];
    EXPECT_EQ(t.material(), m_ptr.get()) << "Pointers to material differ, connection wrong";
}

TEST(obj_io, DISABLED_loading_complex) {
    world_geometry w;
    w.load("mini_cooper.obj");

    EXPECT_EQ(w.vertex_count(), 234435) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 304135) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 49) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 15) << "Bad number of materials";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
