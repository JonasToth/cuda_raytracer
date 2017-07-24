#include "gtest/gtest.h"

#include "graphic/shading.h"
#include "graphic/vector.h"


TEST(shading, ambient_coeff)
{
    EXPECT_FLOAT_EQ(ambient(0.0f, 0.0f), 0.0f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(0.0f, 1.0f), 0.0f) << "Expected no ambient";
    EXPECT_FLOAT_EQ(ambient(1.0f, 0.0f), 0.0f) << "Expected no ambient";

    const auto ExpectedOk = ambient(1.0f, 1.0f);
    EXPECT_FLOAT_EQ(ExpectedOk, 1.0f) << "Expected ambient 1";
    OUT << "Good Value: " << ExpectedOk << std::endl;
}

TEST(shading, diffuse_coeff)
{
    const coord XAxis(1.0f, 0.0f, 0.0f);
    const coord YAxis(0.0f, 1.0f, 0.0f);
    const coord ZAxis(0.0f, 0.0f, 1.0f);

    const coord RealDirection1(normalize(coord(10.0f, 20.0f, -4.0f)));
    const coord RealDirection2(normalize(coord(2.0f, 1.0f, -6.0f)));

    const float MatNone = 0.0f;
    const float LightNone = 0.0f;

    const float MatGud = 0.5f;
    const float LightGud = 1.6f;

    const float dot_product = dot(RealDirection1, RealDirection2);

    // Playing with the material coefficients
    EXPECT_FLOAT_EQ(diffuse(MatNone, LightNone, dot_product), 0.0f)
        << "No diffuse material and light should return 0.0f";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightNone, dot_product), 0.0f)
        << "No diffuse light should return 0.0f";
    EXPECT_FLOAT_EQ(diffuse(MatNone, LightGud, dot_product), 0.0f)
        << "No diffuse material should return 0.0f";
    const auto ExpectedOk = diffuse(MatGud, LightGud, dot_product);
    EXPECT_NE(ExpectedOk, 0.0f) << "diffuse material and light should return !=0.0f";
    OUT << "Good Value: " << ExpectedOk << std::endl;

    // Playing with light source and camera direction
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(XAxis, YAxis)), 0.0f)
        << "Orthogonal directions should be zero";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(XAxis, ZAxis)), 0.0f)
        << "Orthogonal directions should be zero";
    EXPECT_FLOAT_EQ(diffuse(MatGud, LightGud, dot(YAxis, ZAxis)), 0.0f)
        << "Orthogonal directions should be zero";

    EXPECT_NE(diffuse(MatGud, LightGud, dot(RealDirection1, ZAxis)), 0.0f)
        << "Valid Directions result in nonzero diffuse";
}

TEST(shading, specular_coeff)
{
    const coord XAxis(1.0f, 0.0f, 0.0f);
    const coord YAxis(0.0f, 1.0f, 0.0f);
    const coord ZAxis(0.0f, 0.0f, 1.0f);

    const coord RealDirection1(normalize(coord(10.0f, 20.0f, -4.0f)));
    const coord RealDirection2(normalize(coord(2.0f, 1.0f, -6.0f)));

    const float MatNone = 0.0f;
    const float LightNone = 0.0f;

    const float MatGud = 0.5f;
    const float LightGud = 1.6f;

    const float ShineNone = 0.0f;
    const float ShineGud = 2.0f;

    // playing with material coefficients
    float dot_product = dot(RealDirection1, RealDirection2);
    EXPECT_FLOAT_EQ(specular(MatNone, LightNone, dot_product, ShineNone), 0.0f)
        << "No material, light and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightNone, dot_product, ShineGud), 0.0f)
        << "No material, light needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightGud, dot_product, ShineNone), 0.0f)
        << "No material and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatNone, LightGud, dot_product, ShineGud), 0.0f)
        << "No material needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot_product, ShineNone), 0.0f)
        << "No light and shininess needs to result in 0";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot_product, ShineGud), 0.0f)
        << "No light needs to result in 0";
    const auto ExpectedOk = specular(MatGud, LightGud, dot_product, ShineGud);
    EXPECT_NE(ExpectedOk, 0.0f) << "Good Value Expected";
    OUT << "Good Value: " << ExpectedOk << std::endl;

    // playing with directions
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot(XAxis, ZAxis), ShineGud), 0.0f)
        << "Orthogonal directions result in no specular";
    EXPECT_FLOAT_EQ(specular(MatGud, LightNone, dot(YAxis, ZAxis), ShineGud), 0.0f)
        << "Orthogonal directions result in no specular";

    EXPECT_NE(specular(MatGud, LightGud, dot(RealDirection1, ZAxis), ShineGud), 0.0f)
        << "Valid directions result in nonzero";
}

TEST(shading, complex_shade_one_channel)
{
    float spec[3] = {0.5f, 0.5f, 0.5f};
    float diff[3] = {0.5f, 0.5f, 0.5f};
    float ambi[3] = {0.5f, 0.5f, 0.5f};
    const phong_material m(spec, diff, ambi, 1.0f);
    const light_source ls{phong_light(spec, diff), coord(2.0f, 0.0f, 1.0f)};

    const coord P0{0, -1, 1}, P1{-1, 1, 1}, P2{1, 1, 1};
    const coord normal = normalize(cross(P1 - P0, P2 - P1));
    triangle T{&P0, &P1, &P2, &normal};

    const intersect hit(1.0f, coord(0.0f, 0.0f, 0.0f), &T);

    for (float x_dir = 2.0f; x_dir > 0.0f; x_dir -= 0.5f) {
        const auto lv =
            phong_shading(&m, 0.1, normalize(coord(x_dir, 0.0f, -1.0f)), hit, &ls, 1ul,
                          &T, 1ul, flat_shading_tag{}, no_shadow_tag{});
        EXPECT_GT(lv.r, 0.0f) << "Light must be there";
        EXPECT_GT(lv.g, 0.0f) << "Light must be there";
        EXPECT_GT(lv.b, 0.0f) << "Light must be there";
        EXPECT_FLOAT_EQ(lv.r, lv.g) << "Expect same values for all channels";
        EXPECT_FLOAT_EQ(lv.r, lv.b) << "Expect same values for all channels";
        std::clog << "Light Value = (" << lv.r << ", " << lv.g << ", " << lv.b << ")"
                  << std::endl;
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
