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

    const float MatNone   = 0.0f;
    const float LightNone = 0.0f;

    const float MatGud   = 0.5f;
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

    const float MatNone   = 0.0f;
    const float LightNone = 0.0f;

    const float MatGud   = 0.5f;
    const float LightGud = 1.6f;

    const float ShineNone = 0.0f;
    const float ShineGud  = 2.0f;

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
            phong_shading(&m, 0.1, normalize(coord(x_dir, 0.0f, -1.0f)), hit, {&ls, 1ul},
                          {&T, 1ul}, flat_shading_tag{}, no_shadow_tag{});
        EXPECT_GT(lv.r, 0.0f) << "Light must be there";
        EXPECT_GT(lv.g, 0.0f) << "Light must be there";
        EXPECT_GT(lv.b, 0.0f) << "Light must be there";
        EXPECT_FLOAT_EQ(lv.r, lv.g) << "Expect same values for all channels";
        EXPECT_FLOAT_EQ(lv.r, lv.b) << "Expect same values for all channels";
        std::clog << "Light Value = (" << lv.r << ", " << lv.g << ", " << lv.b << ")"
                  << std::endl;
    }
}

TEST(shading, complex_with_shadow)
{
    float spec[3] = {0.0f, 0.0f, 0.0f};
    float diff[3] = {1.0f, 0.0f, 0.0f};
    float ambi[3] = {0.0f, 0.0f, 0.0f};

    const phong_material m{spec, diff, ambi, 1.0f};
    const light_source L[2] = {
        light_source{phong_light(spec, diff), coord(0.0f, 5.0f, 0.0f)},
        light_source{phong_light(spec, diff), coord(-1.0f, 5.0f, 0.0f)}};

    const coord P0{1.0f, 2.0f, 1.0f}, P1{1.0f, 2.0f, -1.0f}, P2{-1.0f, 2.0f, 1.0f};
    const coord P3{1.0f, 0.0f, -1.0f}, P4{-1.0f, 0.0f, -1.0f}, P5{1.0f, 0.0f, 1.0f};
    const coord normal{0.0f, 1.0f, 0.0f};

    const triangle T[2] = {triangle{&P0, &P1, &P2, &normal},
                           triangle{&P3, &P4, &P5, &normal}};

    const coord camera_origin(-3.0f, 3.0f, 0.0f);

    const coord ray_destination[3] = {
        coord{0.0f, 0.0f, -0.5f}, // no shadow
        coord{0.8f, 0.0f, -0.5f}, // half shadow
        coord{0.8f, 0.0f, 0.5f},  // full shadow
    };

    const ray rays[3] = {
        ray(camera_origin, normalize(ray_destination[0] - camera_origin)), // no shadow
        ray(camera_origin, normalize(ray_destination[1] - camera_origin)), // half shadow
        ray(camera_origin, normalize(ray_destination[2] - camera_origin)), // full shadow
    };

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(rays[i].intersects(T[1]).first)
            << "Expect intersection with triangle 1";
    }

    // no shadow point
    const auto no_shadow_color =
        phong_shading(&m, 0.0, rays[0].direction, rays[0].intersects(T[1]).second,
                      {L, 2ul}, {T, 2ul}, flat_shading_tag{}, hard_shadow_tag{});

    const auto half_shadow_color =
        phong_shading(&m, 0.0, rays[1].direction, rays[1].intersects(T[1]).second,
                      {L, 2ul}, {T, 2ul}, flat_shading_tag{}, hard_shadow_tag{});

    const auto full_shadow_color =
        phong_shading(&m, 0.0, rays[2].direction, rays[2].intersects(T[1]).second,
                      {L, 2ul}, {T, 2ul}, flat_shading_tag{}, hard_shadow_tag{});

    EXPECT_GT(no_shadow_color.r, half_shadow_color.r)
        << "No shadow need higher intensity";
    EXPECT_GT(half_shadow_color.r, full_shadow_color.r)
        << "No shadow need higher intensity";
    EXPECT_EQ(full_shadow_color.r, 0) << "Full shadown yields to black";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
