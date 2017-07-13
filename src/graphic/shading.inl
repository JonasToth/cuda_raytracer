#include "graphic/shading.h"

inline CUCALL color phong_shading(const phong_material* m, const float ambient_constant,
                                  const light_source* lights, std::size_t light_count,
                                  const coord& ray_direction, const intersect& hit)
{
    const auto N = normalize(hit.face->normal());
    const auto V = normalize(coord(ray_direction.x, ray_direction.y, ray_direction.z));

    color c{0.f, 0.f, 0.f};

    const auto& mr = m->r;
    const auto& mg = m->g;
    const auto& mb = m->b;

    // currently zero, since no global ambient coefficient for all lights
    c.r = ambient(mr.ambient_reflection(), ambient_constant);
    c.g = ambient(mg.ambient_reflection(), ambient_constant);
    c.b = ambient(mb.ambient_reflection(), ambient_constant);
    
    for(std::size_t i = 0; i < light_count; ++i)
    {
        const auto& lr = lights[i].light.r;
        const auto& lg = lights[i].light.g;
        const auto& lb = lights[i].light.b;

        // Vector zu Licht
        const auto L = normalize(lights[i].position - hit.hit);
        // Reflectionsrichtung des Lichts
        const auto R = normalize(2 * dot(L, N) * N - L);

        {
        const float dot_product = dot(N, L);
        if(dot_product > 0.f)
        {
            c.r+= diffuse(mr.diffuse_reflection(), lr.diffuse_color(), dot_product);
            c.g+= diffuse(mg.diffuse_reflection(), lg.diffuse_color(), dot_product);
            c.b+= diffuse(mb.diffuse_reflection(), lb.diffuse_color(), dot_product);
        }
        }

        {
        const float dot_product = dot(R, V);
        if(dot_product > 0.f)
        {
            c.r+= specular(mr.specular_reflection(), lr.specular_color(), dot_product, 
                           m->shininess());
            c.g+= specular(mg.specular_reflection(), lg.specular_color(), dot_product, 
                           m->shininess());
            c.b+= specular(mb.specular_reflection(), lb.specular_color(), dot_product, 
                           m->shininess());
        }
        }
    }

    return c;

}
