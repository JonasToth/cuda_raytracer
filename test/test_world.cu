#include "gtest/gtest.h"

#include "graphic/material.h"
#include "management/world.h"

#include <algorithm>
#include <vector>
#include "thrust/logical.h"

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

struct validityCheck {
    CUCALL bool operator()(const triangle& t) { return t.isValid(); }
};

TEST(cube, all_contained) 
{
    world_geometry w("cube.obj");
    
    EXPECT_EQ(w.vertex_count(), 8) << "Bad Number of Vertices";
    EXPECT_EQ(w.normal_count(), 6) << "Bad Number of Normals";
    EXPECT_EQ(w.triangle_count(), 12) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 1) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 1) << "Bad number of materials";

    const thrust::host_vector<coord> h_vertices = w.vertices();
    const std::vector<coord> vertices(h_vertices.begin(), h_vertices.end());

    EXPECT_EQ(vertices[0], coord(1.f,       -1.f, -1.f));
    EXPECT_EQ(vertices[1], coord(1.f,       -1.f,  1.f));
    EXPECT_EQ(vertices[2], coord(-1.f,      -1.f,  1.f));
    EXPECT_EQ(vertices[3], coord(-1.f,      -1.f, -1.f));
    EXPECT_EQ(vertices[4], coord(1.f,        1.f, -0.9999999f));
    EXPECT_EQ(vertices[5], coord(0.999999f,  1.f,  1.000001f));
    EXPECT_EQ(vertices[6], coord(-1.f,       1.f,  1.f));
    EXPECT_EQ(vertices[7], coord(-1.f,       1.f, -1.f));

    const thrust::host_vector<coord> h_normals = w.normals();
    const std::vector<coord> normals(h_normals.begin(), h_normals.end());

    EXPECT_EQ(normals[0], coord( 0.f, -1.f,  0.f)) << "Bad normal";
    EXPECT_EQ(normals[1], coord( 0.f,  1.f,  0.f)) << "Bad normal";
    EXPECT_EQ(normals[2], coord( 1.f,  0.f,  0.f)) << "Bad normal";
    EXPECT_EQ(normals[3], coord( 0.f,  0.f,  1.f)) << "Bad normal";
    EXPECT_EQ(normals[4], coord(-1.f,  0.f,  0.f)) << "Bad normal";
    EXPECT_EQ(normals[5], coord( 0.f,  0.f, -1.f)) << "Bad normal";

    const thrust::host_vector<triangle> h_triangles = w.triangles();
    const std::vector<triangle> triangles(h_triangles.begin(), h_triangles.end());

    // correct
    EXPECT_EQ(&triangles[0].p0(), (&w.vertices()[1]).get()) << "Bad triangle connection";
    EXPECT_EQ(&triangles[0].p1(), (&w.vertices()[3]).get()) << "Bad triangle connection";
    EXPECT_EQ(&triangles[0].p2(), (&w.vertices()[0]).get()) << "Bad triangle connection";

    EXPECT_EQ(&triangles[0].normal(), (&w.normals()[0]).get()) << "Bad normal connection";

    EXPECT_NE(triangles[0].material(), nullptr) << "No material connection";

    // incorrect should be checked as well, random choice
    EXPECT_NE(&triangles[0].p0(), (&w.vertices()[0]).get()) << "Bad triangle connection";
    EXPECT_NE(&triangles[0].p1(), (&w.vertices()[5]).get()) << "Bad triangle connection";
    EXPECT_NE(&triangles[0].p2(), (&w.vertices()[2]).get()) << "Bad triangle connection";
}

TEST(cube, vertex_normals)
{
    world_geometry w("cube_subdiv_1.obj");
    
    EXPECT_EQ(w.vertex_count(), 98)    << "Bad Number of Vertices";
    EXPECT_EQ(w.normal_count(), 290)   << "Bad Number of Normals";
    EXPECT_EQ(w.triangle_count(), 192) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 1)      << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 1)   << "Bad number of materials";

    const thrust::host_vector<coord> h_vertices = w.vertices();
    const std::vector<coord> vertices(h_vertices.begin(), h_vertices.end());

    EXPECT_EQ(vertices[0], coord(-0.753587f,  0.320810f, -0.329906f));
    EXPECT_EQ(vertices[1], coord(-0.861419f,  0.002644f, -0.011739f));
    EXPECT_EQ(vertices[2], coord(-0.796206f,  0.369578f, -0.011739f));

    const thrust::host_vector<coord> h_normals = w.normals();
    const std::vector<coord> normals(h_normals.begin(), h_normals.end());

    EXPECT_EQ(normals[0], coord(-0.8774f,  0.3392f, -0.3392f));
    EXPECT_EQ(normals[1], coord(-1.0000f,  0.0000f, -0.0000f));
    EXPECT_EQ(normals[2], coord(-0.9311f,  0.3647f, -0.0000f));

    const thrust::host_vector<triangle> h_triangles = w.triangles();
    const std::vector<triangle> triangles(h_triangles.begin(), h_triangles.end());

    std::clog << "First normal address: " << (&w.normals()[0]).get() << '\n'
              << "Last normal address:  " << (&w.normals()[289]).get() << std::endl;
    // correct
    EXPECT_EQ(&triangles[0].p0(), (&w.vertices()[0]).get()) << "Bad triangle connection";
    EXPECT_EQ(&triangles[0].p1(), (&w.vertices()[1]).get()) << "Bad triangle connection";
    EXPECT_EQ(&triangles[0].p2(), (&w.vertices()[2]).get()) << "Bad triangle connection";

    EXPECT_EQ(&triangles[0].p0_normal(), (&w.normals()[0]).get()) << "Bad normal connection";
    EXPECT_EQ(&triangles[0].p1_normal(), (&w.normals()[1]).get()) << "Bad normal connection";
    EXPECT_EQ(&triangles[0].p2_normal(), (&w.normals()[2]).get()) << "Bad normal connection";

    EXPECT_NE(triangles[0].material(), nullptr) << "unexpected material connection";

    // incorrect should be checked as well, random choice
    EXPECT_NE(&triangles[0].p0(), (&w.vertices()[1]).get()) << "Bad triangle connection";
    EXPECT_NE(&triangles[0].p1(), (&w.vertices()[5]).get()) << "Bad triangle connection";
    EXPECT_NE(&triangles[0].p2(), (&w.vertices()[3]).get()) << "Bad triangle connection";
}

TEST(cube, no_normals) 
{
    world_geometry w("cube_no_normals.obj");

    EXPECT_EQ(w.vertex_count(), 8) << "Bad Number of Vertices";

    // computed while loading for each face
    EXPECT_EQ(w.normal_count(), 12) << "Bad Number of Normals"; 

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

TEST(cube, no_normals_no_materials)
{
    world_geometry w("cube_no_normals_no_materials.obj");

}

TEST(real, easy_scene) 
{
    world_geometry w;
    w.load("shapes.obj");

    EXPECT_EQ(w.vertex_count(), 122) << "Bad Number of Vertices";
    EXPECT_EQ(w.normal_count(), 126) << "Bad Number of Normals";
    EXPECT_EQ(w.triangle_count(), 228) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 4) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 0) << "Bad number of materials";

    const bool isValidRange = thrust::all_of(w.triangles().begin(), w.triangles().end(), 
                                             validityCheck());
    std::clog << "Tested on GPU" << std::endl;
    EXPECT_EQ(isValidRange, true) << "Invalid triangles found";
}

TEST(real, test_bad_input) 
{
    world_geometry w;

    w.load("bad.obj");
    //ASSERT_THROW(w.load("bad.obj"), std::invalid_argument) << "Did not notice the quad";
}


TEST(real, simple_scene)
{
    world_geometry w("material_scene.obj");
    EXPECT_EQ(w.vertex_count(), 8500) << "Bad Number of Vertices";
    EXPECT_EQ(w.normal_count(), 11040) << "Bad Number of Normals";
    EXPECT_EQ(w.triangle_count(), 16976) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 5) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 4) << "Bad number of materials";
}

TEST(real, DISABLED_massive_scene) {
    world_geometry w;
    w.load("mini_cooper.obj");

    EXPECT_EQ(w.vertex_count(), 234435) << "Bad Number of Vertices";
    EXPECT_EQ(w.normal_count(), 347377) << "Bad Number of Normals";
    EXPECT_EQ(w.triangle_count(), 304135) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 49) << "Bad number of Shapes";
    EXPECT_EQ(w.material_count(), 15) << "Bad number of materials";
}

int main(int argc, char** argv)
{
    world_geometry w("cube_no_normals.obj");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
