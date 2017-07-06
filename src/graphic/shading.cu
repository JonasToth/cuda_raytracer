#include "graphic/shading.h"

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
