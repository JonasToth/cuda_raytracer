#include "graphic/kernels/trace.h"
#include "graphic/kernels/image_loop_macro.h"

void trace_single_triangle(memory_surface& surface, const triangle& t)
{
    const float focal_length = 1.f;
    const auto width         = surface.width();
    const auto height        = surface.height();

    PIXEL_LOOP(surface)
    {
        ray r;
        r.origin    = coord{0.f, 0.f, -1.f};
        float dx    = 2.f / ((float)width - 1);
        float dy    = 2.f / ((float)height - 1);
        r.direction = coord{x * dx - 1.f, y * dy - 1.f, focal_length};

        pixel_rgba pixel_color;
        pixel_color.r = 255;
        pixel_color.g = 255;
        pixel_color.b = 255;
        pixel_color.a = 255;

        const auto traced = r.intersects(t);

        if (traced.first)
            surface.write_pixel(x, y, pixel_color);
    }
}


template <typename Camera>
void trace_many_triangles_with_camera(memory_surface& surface, Camera c,
                                      gsl::span<const triangle> triangles)
{
    PIXEL_LOOP(surface)
    {
        ray r = c.rayAt(x, y);

        pixel_rgba pixel_color;
        pixel_color.r = 255;
        pixel_color.g = 255;
        pixel_color.b = 255;
        pixel_color.a = 255;

        triangle const* nearest = nullptr;
        intersect nearest_hit;
        const auto result_pair = calculate_intersection(r, triangles);
        nearest                = result_pair.first;
        nearest_hit            = result_pair.second;

        if (nearest != nullptr) {
            pixel_color.r = nearest_hit.depth * 5.f;
            pixel_color.g = nearest_hit.depth * 5.f;
            pixel_color.b = nearest_hit.depth * 5.f;
            surface.write_pixel(x, y, pixel_color);
        }
    }
}
