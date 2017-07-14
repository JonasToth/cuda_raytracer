#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graphic/kernels/trace.h"
#include "management/window.h"
#include "management/world.h"
#include "util/kernel_launcher/world_shading.h"

#include <thread>
#include <chrono>


void raytrace_many_cuda(cudaSurfaceObject_t& Surface, 
                        const camera& c,
                        const triangle* Triangles,
                        int TriangleCount) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    trace_many_triangles_with_camera<<<dimGrid, dimBlock>>>(Surface, c, 
                                                            Triangles, TriangleCount, 
                                                            c.width(), c.height());
}



auto BM_CubeRender = [](benchmark::State& state, std::string base_name)
{
    window win(800, 600, base_name);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.0f, 2.0f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene(base_name + ".obj");

    state.counters["vertices"]  = scene.vertex_count();
    state.counters["normals"]   = scene.normal_count();
    state.counters["triangles"] = scene.triangle_count();
    state.counters["lights"]    = 1;

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {0.8f, 0.9f, 1.5f});

    const auto& triangles = scene.triangles();

    while(state.KeepRunning())
    {
        raytrace_many_shaded(render_surface.getSurface(), scene.handle());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png(base_name + ".png");
};

auto BM_CubeDepth = [](benchmark::State& state, std::string base_name)
{
    const std::string prefix = "depth_";

    window win(800, 600, base_name);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.0f, 2.0f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene(base_name + ".obj");

    state.counters["vertices"]  = scene.vertex_count();
    state.counters["normals"]   = scene.normal_count();
    state.counters["triangles"] = scene.triangle_count();

    const auto& triangles = scene.triangles();

    while(state.KeepRunning())
    {
        raytrace_many_cuda(render_surface.getSurface(), c,
                           triangles.data().get(), triangles.size());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png(prefix + base_name + ".png");
};

int main(int argc, char** argv)
{
    for(auto& name: {"cube_subdiv_1", "cube_subdiv_2", "cube_subdiv_3", "cube_subdiv_4"}) 
                   /*,"cube_subdiv_5", "cube_subdiv_6"})*/
    {
        auto* b = benchmark::RegisterBenchmark(name, BM_CubeRender, name);
        b->Unit(benchmark::kMicrosecond);
        b->MinTime(0.8);
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
