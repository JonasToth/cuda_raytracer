#include "gtest/gtest.h"
#include "triangle.h"

TEST(triangle_test, construction)
{
    triangle t1;
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
