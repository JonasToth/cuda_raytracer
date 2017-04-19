// Testcase to load an object file

#include "gtest/gtest.h"


TEST(ad_san, test_out_of_bounds) {
    char some_array[10];
    some_array[15] = 'c';
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
