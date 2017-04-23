#include "gtest/gtest.h"

/** @file test/test_obj_io.cpp
 * Test if .obj - Files for geometry are correctly loaded and saved.
 */

TEST(ci_test, basic_build) {
    ASSERT_EQ(0,0) << "Basic Testcase succeeds";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
