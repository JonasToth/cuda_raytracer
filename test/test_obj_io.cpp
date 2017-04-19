// Testcase to load an object file

#include "gtest/gtest.h"


TEST(ci_test, basic_build) {
    ASSERT_EQ(0,0) << "Basic Testcase succeeds";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
