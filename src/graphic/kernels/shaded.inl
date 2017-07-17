template <typename ShadingStyleTag>
__global__ void trace_triangles_shaded(cudaSurfaceObject_t surface, camera c,
                                       const triangle* triangles, std::size_t n_triangles,
                                       const light_source* lights, std::size_t n_lights,
                                       ShadingStyleTag sst)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < c.width() && y < c.height()) {
        ray r = c.rayAt(x, y);

        uchar4 pixel_color;
        pixel_color.x = 0;
        pixel_color.y = 0;
        pixel_color.z = 255;
        pixel_color.w = 255;

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
            const phong_material* hit_material = nearest->material();
            const auto color = phong_shading(hit_material, 0.1, lights, n_lights,
                                             normalize(r.direction), nearest_hit, sst);

            pixel_color.x = 255 * clamp(0.f, color.r, 1.f);
            pixel_color.y = 255 * clamp(0.f, color.g, 1.f);
            pixel_color.z = 255 * clamp(0.f, color.b, 1.f);

            surf2Dwrite(pixel_color, surface, x * sizeof(pixel_color), y);
        }
    }
}
