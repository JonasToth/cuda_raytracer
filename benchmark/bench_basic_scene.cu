#include "benchmark/benchmark.h"


static void BM_SceneRender(benchmark::State& state)
{
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene("material_scene.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    float ambi[3] = {0.2f, 0.2f, 0.2f};
    float no_ambi[3] = {0.01f, 0.01f, 0.01f};
    thrust::device_vector<light_source> lights;
    lights.push_back({{spec, diff, ambi}, {0.8f, 0.9f, 1.5f}});
    lights.push_back({{spec, diff, no_ambi}, {1.7f, -1.1f, -0.3f}});
    lights.push_back({{spec, diff, no_ambi}, {-1.3f, 0.8f, 2.0f}});
    lights.push_back({{spec, diff, no_ambi}, {-1.7f, -1.7f, 0.8f}});

    const auto& triangles = scene.triangles();
    raytrace_many_shaded(render_surface.getSurface(), c,
                         triangles.data().get(), triangles.size(),
                         lights.data().get(), lights.size());

}

BENCHMARK(BM_SceneRender);

BENCHMARK_MAIN()
