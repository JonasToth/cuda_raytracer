#include "gtest/gtest.h"

#include "graphic/kernels/utility.h"
#include "graphic/kernels/trace.h"
#include "graphic/kernels/shaded.h"
#include "macros.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "obj_io.h"

#include <iostream>

const int Width = 800, Height = 800;
camera c(Width, Height, {-2.f, 1.f, -2.f}, {2.f, -1.f, 2.f});

double m_x = 0., m_y = 0.;

static void handle_keys(GLFWwindow* w)
{
    const float dP     = 0.1;
    const float dAngle = M_PI / 180. * 5.;

    const auto& im = input_manager::instance();

    if(im.isPressed(GLFW_KEY_ESCAPE))
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    else if(im.isPressed(GLFW_KEY_A))
        c.move({-dP, 0.f, 0.f});
    else if(im.isPressed(GLFW_KEY_D))
        c.move({dP, 0.f, 0.f});
    else if(im.isPressed(GLFW_KEY_W))
        c.move({0.f, 0.f, dP});
    else if(im.isPressed(GLFW_KEY_S))
        c.move({0.f, 0.f, -dP});
    else if(im.isPressed(GLFW_KEY_Q))
        c.move({0.f, dP, 0.f});
    else if(im.isPressed(GLFW_KEY_E))
        c.move({0.f, -dP, 0.f});
    else if(im.isPressed(GLFW_KEY_LEFT))
        c.swipe(0.f, -dAngle, 0.f);
    else if(im.isPressed(GLFW_KEY_RIGHT))
        c.swipe(0.f, dAngle, 0.f);
    else if(im.isPressed(GLFW_KEY_UP))
        c.swipe(dAngle, 0.f, 0.f);
    else if(im.isPressed(GLFW_KEY_DOWN))
        c.swipe(-dAngle, 0.f, 0.f);
    else
        return;

    std::clog << "Camera Position: " << c.origin() << std::endl;
    std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
}

static void handle_mouse_movement()
{
    const auto& im = input_manager::instance();

    double beta   = -2. * M_PI * im.mouse_diff_x() / Width * 0.01;
    double gamma_ = M_PI * im.mouse_diff_y() / Height * 0.1;
    c.swipe(0., beta, gamma_);
}

void invokeRenderingKernel(cudaSurfaceObject_t& Surface, float t)
{
    //std::clog << "Rendering new image " << char{t} << std::endl;
    dim3 dimBlock(32,32);
    dim3 dimGrid((Width  + dimBlock.x) / dimBlock.x,
                 (Height + dimBlock.y) / dimBlock.y);
    stupid_colors<<<dimGrid, dimBlock>>>(Surface, Width, Height, t);
}

TEST(cuda_draw, basic_drawing) {
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwMakeContextCurrent(w);

    surface_raii vis(Width, Height);

    float t = 0.f;
    while(!glfwWindowShouldClose(w)) {
        t += 0.5f;
        invokeRenderingKernel(vis.getSurface(), t);

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwPollEvents();
        handle_keys(w);
    }
    input_manager::instance().clear();
}

void raytrace_cuda(cudaSurfaceObject_t& Surface, const triangle* T) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((Width + dimBlock.x) / dimBlock.x,
                 (Height+ dimBlock.y) / dimBlock.y);
    trace_single_triangle<<<dimGrid, dimBlock>>>(Surface, T, Width, Height);
}


void raytrace_many_cuda(cudaSurfaceObject_t& Surface, 
                        const camera& c,
                        const triangle* Triangles,
                        int TriangleCount) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(Surface, c.width(), c.height());
    trace_many_triangles_with_camera<<<dimGrid, dimBlock>>>(Surface, c, 
                                                            Triangles, TriangleCount, 
                                                            c.width(), c.height());
}

void raytrace_many_shaded(cudaSurfaceObject_t& surface, camera c,
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

TEST(cuda_draw, drawing_traced_triangle) 
{
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwMakeContextCurrent(w);

    surface_raii vis(Width, Height);

    // Create the Triangle and Coordinates on the device
    thrust::device_vector<coord> Vertices(5);
    Vertices[0] = {0,-1,1}; 
    Vertices[1] = {-1,1,1};
    Vertices[2] = {1,1,1};
    Vertices[3] = {1,-0.8,1};
    Vertices[4] = {-1,0.8,1};

    const auto P0 = Vertices[0];
    const auto P1 = Vertices[1];
    const auto P2 = Vertices[2];
    const auto P3 = Vertices[3];
    const auto P4 = Vertices[4];

    thrust::device_vector<triangle> Triangles(3);
    Triangles[0] = {P0, P1, P2};
    Triangles[1] = {P0, P1, P3};
    Triangles[2] = {P4, P2, P0};

    while(!glfwWindowShouldClose(w)) {
        dim3 dimBlock(32,32);
        dim3 dimGrid((Width + dimBlock.x) / dimBlock.x,
                     (Height+ dimBlock.y) / dimBlock.y);
        black_kernel<<<dimGrid, dimBlock>>>(vis.getSurface(), Width, Height);

        for(std::size_t i = 0; i < Triangles.size(); ++i)
        {
            const thrust::device_ptr<triangle> T = &Triangles[i];
            raytrace_cuda(vis.getSurface(), T.get());
        }

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents(); handle_keys(w); } 
    input_manager::instance().clear();
}

TEST(cuda_draw, draw_loaded_geometry)
{
    // Window stuff
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwSetCursorPosCallback(w, register_mouse_movement);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwMakeContextCurrent(w);

    // Cuda stuff
    surface_raii vis(Width, Height);

    // 3D Stuff
    world_geometry world("shapes.obj");

    const auto& Triangles = world.triangles();

    while(!glfwWindowShouldClose(w)) {
        dim3 dimBlock(32,32);
        dim3 dimGrid((Width + dimBlock.x) / dimBlock.x,
                     (Height + dimBlock.y) / dimBlock.y);
        black_kernel<<<dimGrid, dimBlock>>>(vis.getSurface(), Width, Height);

        raytrace_many_cuda(vis.getSurface(), c, 
                           Triangles.data().get(), Triangles.size());

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents();
        handle_keys(w);
        handle_mouse_movement();
    } 
    input_manager::instance().clear();
}

TEST(cuda_draw, draw_phong_shaded)
{
    // Window stuff
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwSetCursorPosCallback(w, register_mouse_movement);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwMakeContextCurrent(w);

    // Cuda stuff
    surface_raii vis(Width, Height);

    // 3D Stuff
    world_geometry world("test_camera_light.obj");

    thrust::device_vector<light_source> lights;
    float spec[3] = {0.2f, 0.4f, 0.2f};
    float diff[3] = {0.8f, 0.6f, 0.7f};
    float ambi[3] = {1.0f, 1.9f, 1.0f};
    //light_source ls = ;
    lights.push_back({{spec, diff, ambi}, {-1.7f, -1.5f, -1.5f}});
    lights.push_back({{spec, diff, ambi}, { 1.3f, -1.8f, -1.2f}});
    //lights.push_back({{spec, diff, ambi}, {-1.1f,  2.0f,  1.1f}});
    //lights.push_back({{spec, diff, ambi}, {-1.5f, -1.5f,  1.5f}});


    const auto& triangles = world.triangles();

    while(!glfwWindowShouldClose(w)) {
        raytrace_many_shaded(vis.getSurface(), c,
                             triangles.data().get(), triangles.size(),
                             lights.data().get(), lights.size());
        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents();
        handle_keys(w);
        handle_mouse_movement();
    } 
    input_manager::instance().clear();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
