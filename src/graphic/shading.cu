#include "graphic/shading.h"

float phong_shading(const phong_material& m,
                    const light_source* lights, std::size_t light_count,
                    const coord& ray_direction, const intersect& hit)
{
    const auto N = hit.normal;
    const auto V = normalize(coord(-ray_direction.x, -ray_direction.y, -ray_direction.z));

    // currently zero, since no global ambient coefficient for all lights
    float value = ambient(m.r.ambient_reflection(), 0.f);

    
    for(std::size_t i = 0; i < light_count; ++i)
    {
        // Vector zu Licht
        const auto L = normalize(lights[i].position - hit.hit);
        // Reflectionsrichtung des Lichts
        const auto R = normalize(2 * dot(L, N) * N - L);

        const auto& light_channel = lights[i].light.r;

        std::clog << "L = " << L << std::endl;
        std::clog << "R = " << R << std::endl;
        std::clog << "N = " << N << std::endl;
        std::clog << "V = " << V << std::endl;
        std::clog << "dot(R,V) = " << dot(R, V) << std::endl;
        
        value+= diffuse(m.r.diffuse_reflection(), lights[i].light.r.diffuse_reflection(),
                        N, L);
        value+= specular(m.r.specular_reflection(), light_channel.specular_reflection(),
                         V, R, m.shininess());
    }
    

    return value;
}
