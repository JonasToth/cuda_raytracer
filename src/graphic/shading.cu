#include "shading.h"

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
        const auto L = normalize(lights[i].position - hit.hit);
        const auto R = normalize(2 * dot(L, N) * N - L);
        
        value+= diffuse(m.r.diffuse_reflection(), lights[i].light.r.diffuse_reflection(),
                        N, L);
        value+= specular(m.r.specular_reflection(), lights[i].light.r.specular_reflection(),
                         V, R, m.shininess());
    }

    return value;
}
