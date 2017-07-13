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

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png(base_name + ".png");
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
