#include "graphic/kernels/trace.h"

void trace_single_triangle(memory_surface& surface, const triangle& T) {
    const float focal_length = 1.f;

    for(std::size_t y = 0; y < surface.height(); ++y)
        for(std::size_t x = 0; x < surface.width(); ++x)
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

            const auto traced = r.intersects(*t);

            if (traced.first)
                surface.write_pixel(x, y, pixel_color);
    }
}


void trace_many_triangles_with_camera(memory_surface& surface, camera c,
                                      const triangle* triangles, int n_triangles)
{
}
