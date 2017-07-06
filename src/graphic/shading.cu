#include "graphic/shading.h"

color phong_shading(const phong_material& m,
                    const light_source* lights, std::size_t light_count,
                    const coord& ray_direction, const intersect& hit)
{
    const auto N = hit.normal;
    const auto V = normalize(coord(-ray_direction.x, -ray_direction.y, -ray_direction.z));

    color c{0.f, 0.f, 0.f};
    // currently zero, since no global ambient coefficient for all lights
    c.r = ambient(m.r.ambient_reflection(), 0.f);
    c.g = ambient(m.g.ambient_reflection(), 0.f);
    c.b = ambient(m.b.ambient_reflection(), 0.f);

    
    for(std::size_t i = 0; i < light_count; ++i)
    {
        // Vector zu Licht
        const auto L = normalize(lights[i].position - hit.hit);
        // Reflectionsrichtung des Lichts
        const auto R = normalize(2 * dot(L, N) * N - L);

        const auto& lr = lights[i].light.r;
        const auto& lg = lights[i].light.g;
        const auto& lb = lights[i].light.b;

        const auto& mr = m.r;
        const auto& mg = m.g;
        const auto& mb = m.b;

        const float dot_product = dot(R, V);

#ifndef __CUDACC__
        std::clog << "L = " << L << std::endl;
        std::clog << "R = " << R << std::endl;
        std::clog << "N = " << N << std::endl;
        std::clog << "V = " << V << std::endl;
        std::clog << "dot(R,V) = " << dot(R, V) << std::endl;
#endif
        
        c.r+= diffuse(mr.diffuse_reflection(), lr.diffuse_reflection(), N, L);
        c.g+= diffuse(mg.diffuse_reflection(), lr.diffuse_reflection(), N, L);
        c.b+= diffuse(mr.diffuse_reflection(), lr.diffuse_reflection(), N, L);

        c.r+= specular(mr.specular_reflection(), lr.specular_reflection(), dot_product, 
                       m.shininess());
        c.g+= specular(mg.specular_reflection(), lg.specular_reflection(), dot_product, 
                       m.shininess());
        c.b+= specular(mb.specular_reflection(), lb.specular_reflection(), dot_product, 
                       m.shininess());
    }

    return c;
}
