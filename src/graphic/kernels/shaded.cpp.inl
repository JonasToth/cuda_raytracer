#include "image_loop_macro.h"
#include <iostream>

template <typename Camera, typename ShadingStyleTag, typename ShadowTag>
void trace_triangles_shaded(memory_surface& surface, Camera c,
                            gsl::span<const triangle> triangles,
                            gsl::span<const light_source> lights, ShadingStyleTag sst,
                            ShadowTag st)
{
    const float ambient_factor = 0.f;
    PIXEL_LOOP(surface)
    {
        ray r = c.rayAt(x, y);

        pixel_rgba pixel_color;
        pixel_color.r = 0;
        pixel_color.g = 0;
        pixel_color.b = 0;
        pixel_color.a = 255;

        const auto result_pair =
            calculate_intersection(r, triangles.data(), triangles.size());
        triangle const* nearest = result_pair.first;
        intersect nearest_hit = result_pair.second;

        if (nearest != nullptr) {
            const phong_material* hit_material = nearest->material();
            const auto color =
                phong_shading(hit_material, ambient_factor, normalize(r.direction),
                              nearest_hit, lights.data(), lights.size(), triangles.data(),
                              triangles.size(), sst, st);

            pixel_color.r = 255 * clamp(0.f, color.r, 1.f);
            pixel_color.g = 255 * clamp(0.f, color.g, 1.f);
            pixel_color.b = 255 * clamp(0.f, color.b, 1.f);

            surface.write_pixel(x, y, pixel_color);
        }
    }
}
