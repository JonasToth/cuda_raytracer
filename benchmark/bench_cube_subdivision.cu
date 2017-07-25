#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graphic/render/depth.h"
#include "graphic/render/shading.h"
#include "management/window.h"
#include "management/world.h"

#include <chrono>
#include <thread>


auto BM_CubeRender = [](benchmark::State& state, std::string base_name) {
    window win(800, 600, base_name);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), {0.0f, 0.0f, 2.0f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene(base_name + ".obj");

    state.counters["vertices"] = scene.vertex_count();
    state.counters["normals"] = scene.normal_count();
    state.counters["triangles"] = scene.triangle_count();
    state.counters["lights"] = 1;

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {0.8f, 0.9f, 1.5f});

    const auto& triangles = scene.triangles();

    while (state.KeepRunning()) {
        render_flat<no_shadow_tag>(render_surface.getSurface(), scene.handle());
    }

    render_surface.render_gl_texture();
    render_surface.save_as_png(base_name + ".png");
};

auto BM_CubeDepth = [](benchmark::State& state, std::string base_name) {
    const std::string prefix = "depth_";

    window win(800, 600, base_name);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), {0.0f, 0.0f, 2.0f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene(base_name + ".obj");

    state.counters["vertices"] = scene.vertex_count();
    state.counters["normals"] = scene.normal_count();
    state.counters["triangles"] = scene.triangle_count();

    const auto& triangles = scene.triangles();

    while (state.KeepRunning()) {
        raytrace_many_cuda(render_surface.getSurface(), c, scene.handle().triangles);
    }

    render_surface.render_gl_texture();
    render_surface.save_as_png(prefix + base_name + ".png");
};

int main(int argc, char** argv)
{
    for (const auto& name : {"cube_subdiv_1", "cube_subdiv_2", "cube_subdiv_3",
                             "cube_subdiv_4", "cube_subdiv_5", "cube_subdiv_6"}) {
        const std::string render_bm_name = std::string(name) + "flat_no_shadow";
        auto* b0 =
            benchmark::RegisterBenchmark(render_bm_name.c_str(), BM_CubeRender, name);
        b0->Unit(benchmark::kMicrosecond);
        b0->MinTime(2.0);

        const std::string depth_bm_name = std::string(name) + "depth";
        auto* b1 = benchmark::RegisterBenchmark(depth_bm_name.c_str(), BM_CubeDepth, name);
        b1->Unit(benchmark::kMicrosecond);
        b1->MinTime(2.0);
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
