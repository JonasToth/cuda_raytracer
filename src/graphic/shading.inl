#include "graphic/shading.h"

ALWAYS_INLINE inline float diffuse(float kd, float id, float dot_product)
{
    return kd * dot_product * id;
}

ALWAYS_INLINE inline float specular(float ks, float is, float dot_product, float alpha)
{
    return ks * std::pow(dot_product, alpha) * is;
}

ALWAYS_INLINE inline coord shading_normal(const triangle& t, coord /* unused */,
                                          flat_shading_tag /* unused */)
{
    return normalize(t.normal());
}

ALWAYS_INLINE inline coord shading_normal(const triangle& t, coord hit,
                                          smooth_shading_tag /* unused */)
{
    return t.interpolated_normal(hit);
}


template <typename ShadingStyleTag, typename ShadowTag>
color phong_shading(const phong_material* m, const float ambient_constant,
                    const coord& ray_direction, const intersect& hit,
                    gsl::span<const light_source> lights,
                    gsl::span<const triangle> triangles, ShadingStyleTag sst, ShadowTag st)
{
    const auto N = shading_normal(*hit.face, hit.hit, sst);
    const auto V = normalize(coord(ray_direction.x, ray_direction.y, ray_direction.z));
    const auto shadow_hit = [&hit, &N]() {
        auto new_hit = hit;
        new_hit.hit  = new_hit.hit + 0.001 * N;
        return new_hit;
    }();

    color c{0.f, 0.f, 0.f};

    const auto& mr = m->r;
    const auto& mg = m->g;
    const auto& mb = m->b;

    // currently zero, since no global ambient coefficient for all lights
    c.r = ambient(mr.ambient_reflection(), ambient_constant);
    c.g = ambient(mg.ambient_reflection(), ambient_constant);
    c.b = ambient(mb.ambient_reflection(), ambient_constant);

    // for (std::size_t i = 0; i < n_lights; ++i) {
    for (const auto& light : lights) {
        if (!luminated_by_light(shadow_hit, light, triangles, st))
            continue;

        const auto& lr = light.light.r;
        const auto& lg = light.light.g;
        const auto& lb = light.light.b;

        // Vector zu Licht
        const auto L = normalize(light.position - hit.hit);
        // Reflectionsrichtung des Lichts
        const auto R = normalize(2 * dot(L, N) * N - L);

        {
            const float dot_product = dot(N, L);
            if (dot_product > 0.f) {
                c.r += diffuse(mr.diffuse_reflection(), lr.diffuse_color(), dot_product);
                c.g += diffuse(mg.diffuse_reflection(), lg.diffuse_color(), dot_product);
                c.b += diffuse(mb.diffuse_reflection(), lb.diffuse_color(), dot_product);
            }
        }

        {
            const float dot_product = dot(R, V);
            if (dot_product > 0.f) {
                c.r += specular(mr.specular_reflection(), lr.specular_color(),
                                dot_product, m->shininess());
                c.g += specular(mg.specular_reflection(), lg.specular_color(),
                                dot_product, m->shininess());
                c.b += specular(mb.specular_reflection(), lb.specular_color(),
                                dot_product, m->shininess());
            }
        }
    }

    return c;
}

inline bool luminated_by_light(const intersect& hit, const light_source& l,
                               gsl::span<const triangle> triangles,
                               hard_shadow_tag /*unused*/)
{
    const auto L      = normalize(l.position - hit.hit);
    const auto origin = hit.hit;

    ray r;
    r.origin    = origin;
    r.direction = L;

#if 0
    const auto result = calculate_intersection(r, triangles, n_triangles);

    if (result.first == nullptr || result.second.depth > norm(l.position - hit.hit))
        return true;

    return false;
#else
    // std::clog << "Shadow Ray with Origin = " << r.origin
    //<< " and direction = " << r.direction << " emitted." << std::endl;
    const auto max_depth = norm(l.position - hit.hit);
    // std::clog << "Max Depth = " << max_depth << std::endl;
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        const auto traced = r.intersects(triangles[i]);
        if (traced.first) {
            // std::clog << "Shadow Ray did hit Triangle" << std::endl;
            if (traced.second.depth < max_depth) {
                // std::clog << "Shadow Ray got absorbed before light source" <<
                // std::endl;
                return false;
            }
        }
    }
    return true;
#endif
}
