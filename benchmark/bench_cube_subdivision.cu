#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graphic/kernels/utility.h"
#include "graphic/kernels/trace.h"
#include "graphic/kernels/shaded.h"
#include "management/window.h"
#include "obj_io.h"

#include <cstdlib>
#include <thread>
#include <chrono>


static void raytrace_many_shaded(cudaSurfaceObject_t& surface, camera c,
                                 const triangle* triangles, std::size_t n_triangles,
                                 const light_source* lights, std::size_t n_lights)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, c.width(), c.height());
    trace_many_triangles_shaded<<<dimGrid, dimBlock>>>(surface, c,
                                                       triangles, n_triangles, 
                                                       lights, n_lights,
                                                       c.width(), c.height());
}



static void BM_CubeRender(benchmark::State& state)
{
    window win(800, 600, "Material Scene", false);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.0f, 2.0f}, {0.01f, 0.f, -1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());
    world_geometry scene("cube_subdiv_1.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    float ambi[3] = {0.2f, 0.2f, 0.2f};
    thrust::device_vector<light_source> lights;
    lights.push_back({phong_light(spec, diff, ambi), {0.8f, 0.9f, 1.5f}});

    const auto& triangles = scene.triangles();

    while(state.KeepRunning())
    {
        raytrace_many_shaded(render_surface.getSurface(), c,
                             triangles.data().get(), triangles.size(),
                             lights.data().get(), lights.size());
    }

    render_surface.render_gl_texture();
    render_surface.save_as_png("cube_subdiv1.png");
}

BENCHMARK(BM_CubeRender)->Unit(benchmark::kMicrosecond)
                        ->MinTime(2.0);

BENCHMARK_MAIN()
