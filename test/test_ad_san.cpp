// Testcase to load an object file

#include "gtest/gtest.h"


TEST(ad_san, test_out_of_bounds) {
    char some_array[10];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warray-bounds"
    some_array[15] = 'c';
#pragma clang diagnostic pop
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
