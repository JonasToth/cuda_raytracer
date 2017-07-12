#include "gtest/gtest.h"

#include "graphic/shading.h"
#include "graphic/vector.h"


TEST(shading, ambient_coeff)
{
    EXPECT_FLOAT_EQ(ambient(0.f, 0.f), 0.f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(0.f, 1.f), 0.f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(1.f, 0.f), 0.f) << "Expected no ambient";

    const auto ExpectedOk = ambient(1.f, 1.f);
    EXPECT_FLOAT_EQ(ExpectedOk, 1.f) << "Expected ambient 1";
    OUT << "Good Value: " << ExpectedOk << std::endl;
}

TEST(shading, diffuse_coeff)
{
    const coord XAxis(1.f, 0.f, 0.f);
    const coord YAxis(0.f, 1.f, 0.f);
    const coord ZAxis(0.f, 0.f, 1.f);

    const coord RealDirection1(normalize(coord(10.f, 20.f, -4.f)));
    const coord RealDirection2(normalize(coord(2.f, 1.f, -6.f)));

    const float MatNone = 0.f;
    const float LightNone = 0.f;

    const float MatGud = 0.5f;
    const float LightGud = 1.6f;

    const float dot_product = dot(RealDirection1, RealDirection2);

    // Playing with the material coefficients
    EXPECT_FLOAT_EQ(diffuse(MatNone, LightNone, dot_product), 0.f) 
                    << "No diffuse material and light should return 0.f";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightNone, dot_product), 0.f) 
                    << "No diffuse light should return 0.f";
    EXPECT_FLOAT_EQ(diffuse(MatNone, LightGud, dot_product), 0.f) 
                    << "No diffuse material should return 0.f";
    const auto ExpectedOk = diffuse(MatGud, LightGud, dot_product);
    EXPECT_NE(ExpectedOk, 0.f) 
              << "diffuse material and light should return !=0.f";
    OUT << "Good Value: " << ExpectedOk << std::endl;

    // Playing with light source and camera direction
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(XAxis, YAxis)), 0.f)
                    << "Orthogonal directions should be zero";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(XAxis, ZAxis)), 0.f)
                    << "Orthogonal directions should be zero";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(YAxis, ZAxis)), 0.f)
                    << "Orthogonal directions should be zero";

    EXPECT_NE(diffuse(MatGud, LightGud, dot(RealDirection1, ZAxis)), 0.f)
                    << "Valid Directions result in nonzero diffuse";
}

TEST(shading, specular_coeff)
{
    const coord XAxis(1.f, 0.f, 0.f);
    const coord YAxis(0.f, 1.f, 0.f);
    const coord ZAxis(0.f, 0.f, 1.f);

    const coord RealDirection1(normalize(coord(10.f, 20.f, -4.f)));
    const coord RealDirection2(normalize(coord(2.f, 1.f, -6.f)));

    const float MatNone   = 0.f;
    const float LightNone = 0.f;

    const float MatGud    = 0.5f;
    const float LightGud  = 1.6f;

    const float ShineNone = 0.f;
    const float ShineGud  = 2.f;

    // playing with material coefficients
    float dot_product = dot(RealDirection1, RealDirection2);
    EXPECT_FLOAT_EQ(specular(MatNone, LightNone, dot_product, ShineNone), 0.f)
                    << "No material, light and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightNone, dot_product, ShineGud), 0.f)
                    << "No material, light needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightGud, dot_product, ShineNone), 0.f)
                    << "No material and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightGud, dot_product, ShineGud), 0.f)
                    << "No material needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot_product, ShineNone), 0.f)
                    << "No light and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot_product, ShineGud), 0.f)
                    << "No light needs to result in 0";
    const auto ExpectedOk = specular(MatGud, LightGud, dot_product, ShineGud);
    EXPECT_NE(ExpectedOk, 0.f) << "Good Value Expected";
    OUT << "Good Value: " << ExpectedOk << std::endl;

    // playing with directions
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot(XAxis, ZAxis), ShineGud), 0.f)
                    << "Orthogonal directions result in no specular";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot(YAxis, ZAxis), ShineGud), 0.f)
                    << "Orthogonal directions result in no specular";

    EXPECT_NE(specular(MatGud, LightGud, dot(RealDirection1, ZAxis), ShineGud), 0.f)
              << "Valid directions result in nonzero";
}

TEST(shading, comple_shade_one_channel)
{
    float spec[3] = {0.5f, 0.5f, 0.5f};
    float diff[3] = {0.5f, 0.5f, 0.5f};
    float ambi[3] = {0.5f, 0.5f, 0.5f};
    const phong_material m(spec, diff, ambi, 1.f);
    const light_source ls = {phong_light(spec, diff, ambi), coord(2.f, 0.0f, 1.f)};

    //const camera c(640, 480, {1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f});
    const intersect hit(1.f, coord(0.f, 0.f, 0.f), coord(0.f, 0.f, 1.f));

    for(float x_dir = 2.f; x_dir > 0.f; x_dir-= 0.5f)
    {
        const auto lv = phong_shading(&m, &ls, 1ul, normalize(coord(x_dir, 0.f, -1.f)), hit);
        EXPECT_GT(lv.r, 0.f) << "Light must be there";
        EXPECT_GT(lv.g, 0.f) << "Light must be there";
        EXPECT_GT(lv.b, 0.f) << "Light must be there";
        EXPECT_FLOAT_EQ(lv.r, lv.g) << "Expect same values for all channels";
        EXPECT_FLOAT_EQ(lv.r, lv.b) << "Expect same values for all channels";
        std::clog << "Light Value = (" 
                  << lv.r << ", " << lv.g << ", " << lv.b
                  << ")" << std::endl;
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
