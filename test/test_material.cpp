#include "gtest/gtest.h"

#include "material.h"

TEST(phong_material, basic_properties)
{
    phong_material m{0.f, 0.f, 0.f, 0.f};

    EXPECT_EQ(m.specular_reflection(), 0.f) << "Wrong specular reflection";
    EXPECT_EQ(m.ks, 0.f) << "Wrong specular reflection";

    EXPECT_EQ(m.diffuse_reflection(), 0.f) << "Wrong diffuse reflection";
    EXPECT_EQ(m.kd, 0.f) << "Wrong diffuse reflection";

    EXPECT_EQ(m.ambient_reflection(), 0.f) << "Wrong ambient reflection";
    EXPECT_EQ(m.ka, 0.f) << "Wrong ambient reflection";
    
    EXPECT_EQ(m.shininess(), 0.f) << "Wrong shininess factor";
    EXPECT_EQ(m.alpha, 0.f) << "Wrong shininess factor";
}

TEST(phong_material, property_change)
{
    phong_material m{0.f, 0.f, 0.f, 0.f};

    m.diffuse_reflection(10.f);
    m.specular_reflection(10.f);
    m.ambient_reflection(10.f);
    m.shininess(10.f);

    EXPECT_EQ(m.specular_reflection(), 10.f) << "Wrong specular reflection";
    EXPECT_EQ(m.ks, 10.f) << "Wrong specular reflection";

    EXPECT_EQ(m.diffuse_reflection(), 10.f) << "Wrong diffuse reflection";
    EXPECT_EQ(m.kd, 10.f) << "Wrong diffuse reflection";

    EXPECT_EQ(m.ambient_reflection(), 10.f) << "Wrong ambient reflection";
    EXPECT_EQ(m.ka, 10.f) << "Wrong ambient reflection";
    
    EXPECT_EQ(m.shininess(), 10.f) << "Wrong shininess factor";
    EXPECT_EQ(m.alpha, 10.f) << "Wrong shininess factor";
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
