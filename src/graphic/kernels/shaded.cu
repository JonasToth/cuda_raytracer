#include "graphic/kernels/shaded.h"

__device__ static unsigned char clamp(float lowest, float value, float highest)
{
    if(value < lowest) 
        return static_cast<unsigned char>(lowest);
    else if(value > highest) 
        return static_cast<unsigned char>(highest);
    else
        return static_cast<unsigned char>(value);
}

__global__ void trace_many_triangles_shaded(cudaSurfaceObject_t surface, camera c,
                                            const triangle* triangles, std::size_t n_triangles,
                                            const light_source* lights, std::size_t n_lights,
                                            int width, int height)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        ray r = c.rayAt(x, y);

        uchar4 pixel_color;
        pixel_color.x = 255;
        pixel_color.y = 255;
        pixel_color.z = 255;
        pixel_color.w = 255;

        triangle const* nearest = nullptr;
        intersect nearest_hit;
        //nearest_hit.depth = std::numeric_limits<float>::max;
        nearest_hit.depth = 10000.f;

        // Find out the closes triangle
        for(std::size_t i = 0; i < n_triangles; ++i)
        {
            const auto traced = r.intersects(triangles[i]);
            if(traced.first)
            {
                if(traced.second.depth < nearest_hit.depth)
                {
                    nearest = &triangles[i];
                    nearest_hit = traced.second;
                }
            }
        }

        if(nearest != nullptr) {
            const auto float_color = phong_shading(*nearest->material(),
                                                   lights, n_lights,
                                                   r.direction, nearest_hit);
            pixel_color.x = clamp(0.f, 255.f * float_color.r, 255.f);
            pixel_color.y = clamp(0.f, 255.f * float_color.g, 255.f);
            pixel_color.z = clamp(0.f, 255.f * float_color.b, 255.f);
            surf2Dwrite(pixel_color, surface, x * 4, y);
        }
    }
}
