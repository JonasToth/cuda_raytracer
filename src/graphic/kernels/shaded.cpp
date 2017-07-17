#include "graphic/kernels/shaded.h"
#include "image_loop_macro.h"
#include <iostream>

template <typename ShadingStyleTag>
void trace_triangles_shaded(memory_surface& surface, camera c,
                            gsl::span<const triangle> triangles,
                            gsl::span<const light_source> lights, ShadingStyleTag sst)
{
    PIXEL_LOOP(surface)
    {
        ray r = c.rayAt(x, y);

        pixel_rgba pixel_color;
        pixel_color.r = 0;
        pixel_color.g = 0;
        pixel_color.b = 255;
        pixel_color.a = 255;

        triangle const* nearest = nullptr;
        intersect nearest_hit;
        // nearest_hit.depth = std::numeric_limits<float>::max;
        nearest_hit.depth = 10000.f;

        // Find out the closes triangle
        for (std::size_t i = 0; i < triangles.size(); ++i) {
            const auto traced = r.intersects(triangles[i]);
            if (traced.first) {
                if (traced.second.depth < nearest_hit.depth) {
                    nearest = &triangles[i];
                    nearest_hit = traced.second;
                }
            }
        }

        if (nearest != nullptr) {
            const phong_material* hit_material = nearest->material();
            const auto color = phong_shading(hit_material, 0.1, lights,
                                             normalize(r.direction), nearest_hit, sst);

            pixel_color.r = 255 * clamp(0.f, color.r, 1.f);
            pixel_color.g = 255 * clamp(0.f, color.g, 1.f);
            pixel_color.b = 255 * clamp(0.f, color.b, 1.f);

            surface.write_pixel(x, y, pixel_color);
        }
    }
}
