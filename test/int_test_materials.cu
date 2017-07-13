#include "graphic/kernels/utility.h"
#include "graphic/kernels/shaded.h"
#include "management/window.h"
#include "obj_io.h"

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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Give the ouputfile as argument, e.g. materials.png" << std::endl;
        return 1;
    }
    window win(800, 600, "Material Scene", false);
    auto w = win.getWindow();
    glfwMakeContextCurrent(w);

    // Camera Setup similar to blender
    camera c(win.getWidth(), win.getHeight(), 
             {0.0f, 0.5f, -2.0f}, {0.1f, 0.f, 1.f});
    surface_raii render_surface(win.getWidth(), win.getHeight());

    std::clog << "Setup Rendering Platform initialized" << std::endl;
    
    world_geometry scene("material_scene.obj");

    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    thrust::device_vector<light_source> lights;
    lights.push_back({phong_light(spec, diff), {-1.4f, -1.4f, -1.4f}});
    lights.push_back({phong_light(spec, diff), { 1.4f, -1.4f, -1.4f}});
    lights.push_back({phong_light(spec, diff), {-1.4f,  1.4f,  1.4f}});
    lights.push_back({phong_light(spec, diff), {-1.4f, -1.4f,  1.4f}});

    std::clog << "World initialized" << std::endl;

    const auto& triangles = scene.triangles();
    raytrace_many_shaded(render_surface.getSurface(), c,
                         triangles.data().get(), triangles.size(),
                         lights.data().get(), lights.size());
    // seems necessary, otherwise the png is empty :/
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png(argv[1]);
    std::clog << "World rendered" << std::endl;

    return 0;
}
