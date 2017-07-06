#include "trace.h"


__global__ void trace_single_triangle(cudaSurfaceObject_t surface, const triangle* t,
                                      std::size_t width, std::size_t height) 
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const float focal_length = 1.f;

    if(x < width && y < height)
    {
        ray r;
        r.origin    = coord{0.f, 0.f, -1.f};
        float dx = 2.f / ((float) width  - 1);
        float dy = 2.f / ((float) height - 1);
        r.direction = coord{x * dx - 1.f, y * dy - 1.f, focal_length};

        uchar4 white;
        white.x = 255;
        white.y = 255;
        white.z = 255;
        white.w = 255;
        
        const auto traced = r.intersects(*t);

        if(traced.first)
            surf2Dwrite(white, surface, x * 4, y);
    }
}


__global__ void trace_many_triangle_with_camera(cudaSurfaceObject_t surface, camera c,
                                                const triangle* triangles, int n_triangles,
                                                int width, int height)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        ray r = c.rayAt(x, y);

        uchar4 white;
        white.x = 255;
        white.y = 255;
        white.z = 255;
        white.w = 255;

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
            white.x = nearest_hit.depth * 5.f;
            white.y = nearest_hit.depth * 5.f;
            white.z = nearest_hit.depth * 5.f;
            surf2Dwrite(white, surface, x * 4, y);
        }
    }
}
