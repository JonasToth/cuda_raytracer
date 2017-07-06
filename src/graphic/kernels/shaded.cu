#include "graphic/kernels/shaded.h"


CUCALL color phong_shading(const phong_material* m,
                           const light_source* lights, std::size_t light_count,
                           const coord& ray_direction, const intersect& hit)
{
    const auto N = normalize(hit.normal);
    const auto V = normalize(coord(ray_direction.x, ray_direction.y, ray_direction.z));

    color c{0.f, 0.f, 0.f};

    const auto& mr = m->r;
    const auto& mg = m->g;
    const auto& mb = m->b;

    // currently zero, since no global ambient coefficient for all lights
    c.r = ambient(mr.ambient_reflection(), 0.5f);
    c.g = ambient(mg.ambient_reflection(), 0.5f);
    c.b = ambient(mb.ambient_reflection(), 0.5f);

    
    for(std::size_t i = 0; i < light_count; ++i)
    {
        // Vector zu Licht
        const auto L = normalize(lights[i].position - hit.hit);
        // Reflectionsrichtung des Lichts
        const auto R = normalize(2 * dot(L, N) * N - L);

        const auto& lr = lights[i].light.r;
        const auto& lg = lights[i].light.g;
        const auto& lb = lights[i].light.b;

        {
        const float dot_product = dot(N, L);
        if(dot_product > 0.f)
        {
            c.r+= diffuse(mr.diffuse_reflection(), lr.diffuse_reflection(), dot_product);
            c.g+= diffuse(mg.diffuse_reflection(), lr.diffuse_reflection(), dot_product);
            c.b+= diffuse(mr.diffuse_reflection(), lr.diffuse_reflection(), dot_product);
        }
        }

        {
        const float dot_product = dot(R, V);
        if(dot_product > 0.f)
        {
            c.r+= specular(mr.specular_reflection(), lr.specular_reflection(), dot_product, 
                           m->shininess());
            c.g+= specular(mg.specular_reflection(), lg.specular_reflection(), dot_product, 
                           m->shininess());
            c.b+= specular(mb.specular_reflection(), lb.specular_reflection(), dot_product, 
                           m->shininess());
        }
        }
    }

    return c;
}


__device__ static float clamp(float lowest, float value, float highest)
{
    if(value < lowest) 
        return lowest;
    else if(value > highest) 
        return highest;
    else
        return value;
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
        pixel_color.x = 0;
        pixel_color.y = 0;
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
            const phong_material* hit_material = nearest->material();
            const auto float_color = phong_shading(hit_material,
                                                   lights, n_lights,
                                                   normalize(r.direction), nearest_hit);

            pixel_color.x = 255 * clamp(0.f, float_color.r, 1.f);
            pixel_color.y = 255 * clamp(0.f, float_color.g, 1.f);
            pixel_color.z = 255 * clamp(0.f, float_color.b, 1.f);

            //pixel_color.x = float_color.r;
            //pixel_color.y = float_color.g;
            //pixel_color.z = float_color.b;

            surf2Dwrite(pixel_color, surface, x * 4, y);
        }
    }
}
