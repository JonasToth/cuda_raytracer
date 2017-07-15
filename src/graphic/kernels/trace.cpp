#include "graphic/kernels/trace.h"
#include "graphic/kernels/image_loop_macro.h"

void trace_single_triangle(memory_surface& surface, const triangle& t)
{
    const float focal_length = 1.f;
    const auto width = surface.width();
    const auto height = surface.height();

    PIXEL_LOOP(surface)
    {
        ray r;
        r.origin = coord{0.f, 0.f, -1.f};
        float dx = 2.f / ((float)width - 1);
        float dy = 2.f / ((float)height - 1);
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


void trace_many_triangles_with_camera(memory_surface& surface, camera c,
                                      const triangle* triangles, std::size_t n_triangles)
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
        // nearest_hit.depth = std::numeric_limits<float>::max;
        nearest_hit.depth = 10000.f;

        // Find out the closes triangle
        for (std::size_t i = 0; i < n_triangles; ++i) {
            const auto traced = r.intersects(triangles[i]);
            if (traced.first) {
                if (traced.second.depth < nearest_hit.depth) {
                    nearest = &triangles[i];
                    nearest_hit = traced.second;
                }
            }
        }

        if (nearest != nullptr) {
            pixel_color.r = nearest_hit.depth * 5.f;
            pixel_color.g = nearest_hit.depth * 5.f;
            pixel_color.b = nearest_hit.depth * 5.f;
            surface.write_pixel(x, y, pixel_color);
        }
    }
}
