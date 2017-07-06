#include "gtest/gtest.h"

#include "material.h"

TEST(phong_material, basic_properties)
{
    float spec[3] = {0.f, 0.f, 0.f};
    float diff[3] = {0.f, 0.f, 0.f};
    float ambi[3] = {0.f, 0.f, 0.f};
    phong_material m(spec, diff, ambi, 0.f);

    EXPECT_FLOAT_EQ(m.r.specular_reflection(), 0.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(m.r.ks, 0.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(m.r.diffuse_reflection(), 0.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(m.r.kd, 0.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(m.r.ambient_reflection(), 0.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(m.r.ka, 0.f) << "Wrong ambient reflection";
    
    EXPECT_FLOAT_EQ(m.shininess(), 0.f) << "Wrong shininess factor";
    EXPECT_FLOAT_EQ(m.alpha, 0.f) << "Wrong shininess factor";
}

TEST(phong_material, property_change)
{
    float spec[3] = {0.f, 0.f, 0.f};
    float diff[3] = {0.f, 0.f, 0.f};
    float ambi[3] = {0.f, 0.f, 0.f};
    phong_material m(spec, diff, ambi, 0.f);

    // Test for red channel, other channels are the same
    m.r.diffuse_reflection(10.f);
    m.r.specular_reflection(10.f);
    m.r.ambient_reflection(10.f);
    m.shininess(10.f);

    EXPECT_FLOAT_EQ(m.r.specular_reflection(), 10.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(m.r.ks, 10.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(m.r.diffuse_reflection(), 10.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(m.r.kd, 10.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(m.r.ambient_reflection(), 10.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(m.r.ka, 10.f) << "Wrong ambient reflection";
    
    EXPECT_FLOAT_EQ(m.shininess(), 10.f) << "Wrong shininess factor";
    EXPECT_FLOAT_EQ(m.alpha, 10.f) << "Wrong shininess factor";
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
