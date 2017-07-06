#include "gtest/gtest.h"

#include "graphic/light.h"

TEST(light, basic_properties)
{
    float spec[3] = {0.f, 0.f, 0.f};
    float diff[3] = {0.f, 0.f, 0.f};
    float ambi[3] = {0.f, 0.f, 0.f};
    phong_light l(spec, diff, ambi);

    // red
    EXPECT_FLOAT_EQ(l.r.specular_reflection(), 0.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(l.r.ks, 0.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(l.r.diffuse_reflection(), 0.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(l.r.kd, 0.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(l.r.ambient_reflection(), 0.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(l.r.ka, 0.f) << "Wrong ambient reflection";
    
    // green
    EXPECT_FLOAT_EQ(l.g.specular_reflection(), 0.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(l.g.ks, 0.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(l.g.diffuse_reflection(), 0.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(l.g.kd, 0.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(l.g.ambient_reflection(), 0.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(l.g.ka, 0.f) << "Wrong ambient reflection";

    // blue
    EXPECT_FLOAT_EQ(l.b.specular_reflection(), 0.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(l.b.ks, 0.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(l.b.diffuse_reflection(), 0.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(l.b.kd, 0.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(l.b.ambient_reflection(), 0.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(l.b.ka, 0.f) << "Wrong ambient reflection";
}

TEST(light, property_change)
{
    float spec[3] = {0.f, 0.f, 0.f};
    float diff[3] = {0.f, 0.f, 0.f};
    float ambi[3] = {0.f, 0.f, 0.f};
    phong_light l(spec, diff, ambi);

    // Test for red channel, other channels are the same
    l.r.diffuse_reflection(10.f);
    l.r.specular_reflection(10.f);
    l.r.ambient_reflection(10.f);

    EXPECT_FLOAT_EQ(l.r.specular_reflection(), 10.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(l.r.ks, 10.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(l.r.diffuse_reflection(), 10.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(l.r.kd, 10.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(l.r.ambient_reflection(), 10.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(l.r.ka, 10.f) << "Wrong ambient reflection";
}

TEST(light_source, basic_properties)
{
    float spec[3] = {10.f, 10.f, 10.f};
    float diff[3] = {10.f, 10.f, 10.f};
    float ambi[3] = {10.f, 10.f, 10.f};
    light_source ls = {{spec, diff, ambi}, {0.f, 0.f, 0.f}};

    EXPECT_FLOAT_EQ(ls.light.r.specular_reflection(), 10.f) << "Wrong specular reflection";
    EXPECT_FLOAT_EQ(ls.light.r.ks, 10.f) << "Wrong specular reflection";

    EXPECT_FLOAT_EQ(ls.light.r.diffuse_reflection(), 10.f) << "Wrong diffuse reflection";
    EXPECT_FLOAT_EQ(ls.light.r.kd, 10.f) << "Wrong diffuse reflection";

    EXPECT_FLOAT_EQ(ls.light.r.ambient_reflection(), 10.f) << "Wrong ambient reflection";
    EXPECT_FLOAT_EQ(ls.light.r.ka, 10.f) << "Wrong ambient reflection";

    EXPECT_FLOAT_EQ(ls.position.x, 0.f) << "Bad x position";
    EXPECT_FLOAT_EQ(ls.position.y, 0.f) << "Bad y position";
    EXPECT_FLOAT_EQ(ls.position.z, 0.f) << "Bad z position";
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
