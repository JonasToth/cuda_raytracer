template <typename Camera, typename ShadingStyleTag, typename ShadowTag>
__global__ void trace_triangles_shaded(cudaSurfaceObject_t surface, Camera c,
                                       const triangle* triangles, std::size_t n_triangles,
                                       const light_source* lights, std::size_t n_lights,
                                       ShadingStyleTag sst, ShadowTag st)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const float ambient_factor = 0.f;

    if (x < c.width() && y < c.height()) {
        ray r = c.rayAt(x, y);

        uchar4 pixel_color;
        pixel_color.x = 0;
        pixel_color.y = 0;
        pixel_color.z = 0;
        pixel_color.w = 255;

        triangle const* nearest = nullptr;
        intersect nearest_hit;
        const auto result_pair = calculate_intersection(r, triangles, n_triangles);
        nearest = result_pair.first;
        nearest_hit = result_pair.second;

        if (nearest != nullptr) {
            const phong_material* hit_material = nearest->material();
            const auto color = phong_shading(hit_material, ambient_factor,
                                             normalize(r.direction), nearest_hit, lights,
                                             n_lights, triangles, n_triangles, sst, st);

            pixel_color.x = 255 * clamp(0.f, color.r, 1.f);
            pixel_color.y = 255 * clamp(0.f, color.g, 1.f);
            pixel_color.z = 255 * clamp(0.f, color.b, 1.f);

            surf2Dwrite(pixel_color, surface, x * sizeof(pixel_color), y);
        }
    }
}
