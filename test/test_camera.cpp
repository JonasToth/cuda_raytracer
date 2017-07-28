#include "gtest/gtest.h"

#include "graphic/camera.h"

TEST(camera, properties)
{
    {
        camera c(640, 480);

        EXPECT_EQ(c.width(), 640) << "Bad width";
        EXPECT_EQ(c.height(), 480) << "Bad height";
        EXPECT_EQ(c.origin(), coord(0.f, 0.f, 0.f)) << "Bad origin";
        EXPECT_EQ(c.steering(), coord(0.f, 0.f, 1.f)) << "Bad steering";
    }

    {
        camera c(640, 480, {1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f});
        EXPECT_EQ(c.width(), 640) << "Bad width";
        EXPECT_EQ(c.height(), 480) << "Bad height";
        EXPECT_EQ(c.origin(), coord(1.f, 1.f, 1.f)) << "Bad origin";
        EXPECT_EQ(c.steering(), normalize(coord(-1.f, -1.f, -1.f))) << "Bad steering";

        c.move({5.f, 0.f, 0.f});
        EXPECT_EQ(c.origin(), coord(6.f, 1.f, 1.f)) << "Bad origin";
    }
}

TEST(camera, ray_generation)
{
    camera c(640, 480);

    auto r = c.rayAt(320, 240);
    ASSERT_EQ(r.origin, coord(0.f, 0.f, 0.f)) << "Not from center of the universe";
    ASSERT_EQ(r.direction, coord(0.f, 0.f, 1.f)) << "Centered ray not in optical axis";

    const auto tl = c.rayAt(0, 0).direction;
    const auto tm = c.rayAt(320, 0).direction;
    const auto tr = c.rayAt(640, 0).direction;

    const auto ml = c.rayAt(0, 240).direction;
    const auto mm = c.rayAt(320, 240).direction;
    const auto mr = c.rayAt(640, 240).direction;

    const auto bl = c.rayAt(0, 480).direction;
    const auto bm = c.rayAt(320, 480).direction;
    const auto br = c.rayAt(640, 480).direction;

    // Rays in one of the coordinate axis
    EXPECT_FLOAT_EQ(tm.x, 0.f) << "x center not centered " << tm;
    EXPECT_FLOAT_EQ(mm.x, 0.f) << "x center not centered " << mm;
    EXPECT_FLOAT_EQ(bm.x, 0.f) << "x center not centered " << bm;

    EXPECT_FLOAT_EQ(ml.y, 0.f) << "y center not centered " << ml;
    EXPECT_FLOAT_EQ(mm.y, 0.f) << "y center not centered " << mm;
    EXPECT_FLOAT_EQ(mr.y, 0.f) << "y center not centered " << mr;

    EXPECT_EQ(mm, coord(0.f, 0.f, 1.f)) << "middle ray not in optical axis " << mm;

    EXPECT_FLOAT_EQ(tl.x, -tr.x) << "x value not mirrored " << tl;
    EXPECT_FLOAT_EQ(ml.x, -mr.x) << "x value not mirrored " << ml;
    EXPECT_FLOAT_EQ(bl.x, -br.x) << "x value not mirrored " << bl;
    EXPECT_LT(tl.x, 0.f) << "left sided ray not on negative x axis " << tl;
    EXPECT_LT(ml.x, 0.f) << "left sided ray not on negative x axis " << ml;
    EXPECT_LT(bl.x, 0.f) << "left sided ray not on negative x axis " << bl;

    EXPECT_FLOAT_EQ(tl.y, -bl.y) << "y value not mirrored " << tl;
    EXPECT_FLOAT_EQ(tm.y, -bm.y) << "y value not mirrored " << tm;
    EXPECT_FLOAT_EQ(tr.y, -br.y) << "y value not mirrored " << tr;
    EXPECT_LT(tl.y, 0.f) << "top rays not on negative y axis " << tl;
    EXPECT_LT(tm.y, 0.f) << "top rays not on negative y axis " << tm;
    EXPECT_LT(tr.y, 0.f) << "top rays not on negative y axis " << tr;

    EXPECT_GT(tl.z, 0.f) << "ray not in positive z direction " << tl;
    EXPECT_GT(tm.z, 0.f) << "ray not in positive z direction " << tm;
    EXPECT_GT(tr.z, 0.f) << "ray not in positive z direction " << tr;
    EXPECT_GT(ml.z, 0.f) << "ray not in positive z direction " << ml;
    EXPECT_GT(mm.z, 0.f) << "ray not in positive z direction " << mm;
    EXPECT_GT(mr.z, 0.f) << "ray not in positive z direction " << mr;
    EXPECT_GT(bl.z, 0.f) << "ray not in positive z direction " << bl;
    EXPECT_GT(bm.z, 0.f) << "ray not in positive z direction " << bm;
    EXPECT_GT(br.z, 0.f) << "ray not in positive z direction " << br;
}

TEST(camera, rays_moved_camera)
{
    // look at center of the world
    camera c(640, 480, {10.f, 10.f, 10.f}, {-10.f, -10.f, -10.f});

    auto r = c.rayAt(320, 240);
    ASSERT_EQ(r.origin, coord(10.f, 10.f, 10.f)) << "Not from center of camera";
    ASSERT_EQ(r.direction, normalize(coord(-1.f, -1.f, -1.f)))
        << "Centered ray not in optical axis";

    std::clog << c.rayAt(0, 0).direction << std::endl;
    std::clog << c.rayAt(640, 0).direction << std::endl;
    std::clog << c.rayAt(0, 480).direction << std::endl;
    std::clog << c.rayAt(640, 480).direction << std::endl;

    c.move({5.f, 0.f, 0.f});
    r = c.rayAt(320, 240);
    ASSERT_EQ(r.origin, coord(15.f, 10.f, 10.f)) << "Not from moved center of camera";
    ASSERT_EQ(r.direction, normalize(coord(-1.f, -1.f, -1.f)))
        << "Centered ray not in optical axis";

    std::clog << c.rayAt(0, 0).direction << std::endl;
    std::clog << c.rayAt(640, 0).direction << std::endl;
    std::clog << c.rayAt(0, 480).direction << std::endl;
    std::clog << c.rayAt(640, 480).direction << std::endl;
}

TEST(equirectengular, ray_generation)
{
    const int width  = 10000;
    const int height = 5000;
    const int cx     = width / 2;
    const int cy     = height / 2;

    // rays on the whole unit sphere
    equirectengular c(width, height);

    auto r = c.rayAt(cx, cy);
    ASSERT_EQ(r.origin, coord(0.f, 0.f, 0.f)) << "Not from center of the universe";
    ASSERT_EQ(r.direction, coord(1.0f, 0.0f, 0.0f)) << "Centered ray not in optical axis";

    const auto tl = c.rayAt(0, 0).direction;
    const auto tm = c.rayAt(cx, 0).direction;
    const auto tr = c.rayAt(width, 0).direction;

    const auto mnorth = c.rayAt(0, cy).direction;
    const auto meast  = c.rayAt(cx / 2, cy).direction;
    const auto msouth = c.rayAt(cx, cy).direction;
    const auto mwest  = c.rayAt(cx + cx / 2, cy).direction;

    const auto bl = c.rayAt(0, height).direction;
    const auto bm = c.rayAt(cx, height).direction;
    const auto br = c.rayAt(width, height).direction;

    std::clog << "TL = " << tl << " "
              << "TM = " << tm << " "
              << "TR = " << tr << '\n'
              << "MN = " << mnorth << " "
              << "ME = " << meast << " "
              << "MS = " << msouth << " "
              << "MW = " << mwest << '\n'
              << "BL = " << bl << " "
              << "BM = " << bm << " "
              << "BR = " << br << std::endl;

    EXPECT_EQ(tl, coord(0.0f, 0.0f, 1.0f));
    EXPECT_EQ(tm, coord(0.0f, 0.0f, 1.0f));
    EXPECT_EQ(tr, coord(0.0f, 0.0f, 1.0f));

    EXPECT_EQ(bl, coord(0.0f, 0.0f, -1.0f));
    EXPECT_EQ(bm, coord(0.0f, 0.0f, -1.0f));
    EXPECT_EQ(br, coord(0.0f, 0.0f, -1.0f));

    EXPECT_EQ(mnorth, coord(-1.0f, 0.0f, 0.0f));
    EXPECT_EQ(meast, coord(0.0f, -1.0f, 0.0f));
    EXPECT_EQ(msouth, coord(1.0f, 0.0f, 0.0f));
    EXPECT_EQ(mwest, coord(0.0f, 1.0f, 0.0f));
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
