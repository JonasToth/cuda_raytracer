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



static void BM_SceneRender(benchmark::State& state)
{
    window win(800, 600, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.5f, 2.5f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene("material_scene.obj");

    state.counters["vertices"]  = scene.vertex_count();
    state.counters["normals"]   = scene.normal_count();
    state.counters["triangles"] = scene.triangle_count();
    state.counters["lights"]    = 4;

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {0.8f, 0.9f, 1.5f});
    scene.add_light(phong_light(spec, diff), {1.7f, -1.1f, -0.3f});
    scene.add_light(phong_light(spec, diff), {-1.3f, 0.8f, 2.0f});
    scene.add_light(phong_light(spec, diff), {-1.7f, -1.7f, 0.8f});

    const auto& triangles = scene.triangles();

    while(state.KeepRunning())
    {
        raytrace_many_shaded(render_surface.getSurface(), scene.handle());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png("material_scene.png");
}

static void BM_SceneDepth(benchmark::State& state)
{
    window win(800, 600, "Material Scene");
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.5f, 2.5f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene("material_scene.obj");

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
    render_surface.save_as_png("material_depth.png");
}

BENCHMARK(BM_SceneRender)->Unit(benchmark::kMicrosecond)
                         ->MinTime(1.0);
BENCHMARK(BM_SceneDepth)->Unit(benchmark::kMicrosecond)
                        ->MinTime(1.0);

BENCHMARK_MAIN()
