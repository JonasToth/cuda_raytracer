#include "gtest/gtest.h"
#include <iostream>

/** @file test/test_ad_san.cpp
 * Test if the Address Sanitizer works and find bugs. Tests Static Analysis silencing as well.
 */

#if (defined (__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
TEST(ad_san, test_out_of_bounds) {
    char some_array[10];
    some_array[9] = 0;

#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warray-bounds"
#endif
    // cppcheck-suppress arrayIndexOutOfBounds
    some_array[15] = 'c'; // NOLINT
#if defined (__clang__)
#pragma clang diagnostic pop
#endif
}
#if (defined (__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic pop
#endif

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
