#include "gtest/gtest.h"

#include "obj_io.h"

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

TEST(obj_io, loading_simple) {
    WorldGeometry w;
    w.load("shapes.obj");

    EXPECT_EQ(w.vertex_count(), 122) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 228) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 4) << "Bad number of Shapes";
}

TEST(obj_io, loading_complex) {
    WorldGeometry w;
    w.load("mini_cooper.obj");

    EXPECT_EQ(w.vertex_count(), 304135) << "Bad Number of Vertices";
    EXPECT_EQ(w.triangle_count(), 234435) << "Bad Number of Triangles";
    EXPECT_EQ(w.shape_count(), 49) << "Bad number of Shapes";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
