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
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
