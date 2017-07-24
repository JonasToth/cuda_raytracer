#ifndef SCENE_SETUP_H_8UCXBVT2
#define SCENE_SETUP_H_8UCXBVT2

void setup_common_scene(world_geometry& scene)
{
    float spec[3] = {0.4f, 0.4f, 0.4f};
    float diff[3] = {0.4f, 0.4f, 0.4f};
    scene.add_light(phong_light(spec, diff), {-5.0f, 10.0f,  5.0f});
    scene.add_light(phong_light(spec, diff), {-5.0f,  4.0f,  5.0f});
    scene.add_light(phong_light(spec, diff), { 5.0f,  4.0f,  5.0f});
    scene.add_light(phong_light(spec, diff), { 5.0f, 10.0f,  5.0f});
}

#endif /* end of include guard: SCENE_SETUP_H_8UCXBVT2 */
