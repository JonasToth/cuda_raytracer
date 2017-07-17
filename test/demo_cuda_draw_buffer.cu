#include "gtest/gtest.h"

#include "graphic/kernels/shaded.h"
#include "graphic/render/world_depth.h"
#include "graphic/render/world_shading.h"
#include "management/input_callback.h"
#include "management/input_manager.h"
#include "management/window.h"
#include "management/world.h"
#include "util/demos/fps_demo.h"

#include <iostream>

const int Width = 800, Height = 800;
camera c(Width, Height, {-2.f, 1.f, -2.f}, {2.f, -1.f, 2.f});

double m_x = 0., m_y = 0.;

static void handle_keys(GLFWwindow* w)
{
    const float dP = 0.1;
    const float dAngle = M_PI / 180. * 5.;

    const auto& im = input_manager::instance();

    if (im.isPressed(GLFW_KEY_ESCAPE))
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    else if (im.isPressed(GLFW_KEY_A))
        c.move({-dP, 0.f, 0.f});
    else if (im.isPressed(GLFW_KEY_D))
        c.move({dP, 0.f, 0.f});
    else if (im.isPressed(GLFW_KEY_W))
        c.move({0.f, 0.f, dP});
    else if (im.isPressed(GLFW_KEY_S))
        c.move({0.f, 0.f, -dP});
    else if (im.isPressed(GLFW_KEY_Q))
        c.move({0.f, dP, 0.f});
    else if (im.isPressed(GLFW_KEY_E))
        c.move({0.f, -dP, 0.f});
    else if (im.isPressed(GLFW_KEY_LEFT))
        c.turn(-dAngle, 0.f);
    else if (im.isPressed(GLFW_KEY_RIGHT))
        c.turn(dAngle, 0.f);
    else if (im.isPressed(GLFW_KEY_UP))
        c.turn(0.f, dAngle);
    else if (im.isPressed(GLFW_KEY_DOWN))
        c.turn(0.f, -dAngle);
    else
        return;

    std::clog << "Camera Position: " << c.origin() << std::endl;
    std::clog << "Camera Steering At: " << c.steering() << std::endl << std::endl;
}

static void handle_mouse_movement()
{
    const auto& im = input_manager::instance();

    double yaw = -2. * M_PI * im.mouse_diff_x() / Width;
    double pitch = M_PI * im.mouse_diff_y() / Height;
    c.turn(yaw, pitch);
}

void invokeRenderingKernel(cudaSurfaceObject_t& Surface, float t)
{
    // std::clog << "Rendering new image " << char{t} << std::endl;
    dim3 dimBlock(32, 32);
    dim3 dimGrid((Width + dimBlock.x) / dimBlock.x, (Height + dimBlock.y) / dimBlock.y);
    stupid_colors<<<dimGrid, dimBlock>>>(Surface, Width, Height, t);
}

TEST(cuda_draw, basic_drawing)
{
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwMakeContextCurrent(w);

    surface_raii vis(Width, Height);

    float t = 0.f;
    while (!glfwWindowShouldClose(w)) {
        t += 0.5f;
        invokeRenderingKernel(vis.getSurface(), t);

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwPollEvents();
        handle_keys(w);
    }
    input_manager::instance().clear();
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
    Vertices[0] = {0, -1, 1};
    Vertices[1] = {-1, 1, 1};
    Vertices[2] = {1, 1, 1};
    Vertices[3] = {1, -0.8, 1};
    Vertices[4] = {-1, 0.8, 1};

    const auto* P0 = (&Vertices[0]).get();
    const auto* P1 = (&Vertices[1]).get();
    const auto* P2 = (&Vertices[2]).get();
    const auto* P3 = (&Vertices[3]).get();
    const auto* P4 = (&Vertices[4]).get();

    thrust::device_vector<coord> Normals(3);
    Normals[0] = normalize(cross(Vertices[1] - Vertices[0], Vertices[2] - Vertices[1]));
    Normals[1] = normalize(cross(Vertices[1] - Vertices[0], Vertices[3] - Vertices[0]));
    Normals[2] = normalize(cross(Vertices[2] - Vertices[4], Vertices[2] - Vertices[0]));

    const auto* t0_n = (&Normals[0]).get();
    const auto* t1_n = (&Normals[1]).get();
    const auto* t2_n = (&Normals[2]).get();

    thrust::device_vector<triangle> Triangles(3);
    Triangles[0] = triangle(P0, P1, P2, t0_n);
    Triangles[1] = triangle(P0, P1, P3, t1_n);
    Triangles[2] = triangle(P4, P2, P0, t2_n);

    while (!glfwWindowShouldClose(w)) {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((Width + dimBlock.x) / dimBlock.x, (Height + dimBlock.y) / dimBlock.y);
        black_kernel<<<dimGrid, dimBlock>>>(vis.getSurface(), Width, Height);

        for (std::size_t i = 0; i < Triangles.size(); ++i) {
            const thrust::device_ptr<triangle> T = &Triangles[i];
            raytrace_cuda(vis.getSurface(), win.getWidth(), win.getHeight(), T.get());
        }

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents();
        handle_keys(w);
    }
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

    c = camera(Width, Height, {-2.f, 1.f, -2.f}, {2.f, -1.f, 2.f});

    // Cuda stuff
    surface_raii vis(Width, Height);

    // 3D Stuff
    world_geometry world("shapes.obj");

    const auto& Triangles = world.triangles();

    while (!glfwWindowShouldClose(w)) {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((Width + dimBlock.x) / dimBlock.x, (Height + dimBlock.y) / dimBlock.y);
        black_kernel<<<dimGrid, dimBlock>>>(vis.getSurface(), Width, Height);

        raytrace_many_cuda(vis.getSurface(), c, Triangles.data().get(), Triangles.size());

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
    c = camera(Width, Height, {-2.f, 1.f, -2.f}, {2.f, -1.f, 2.f});

    glfwSetKeyCallback(w, register_key_press);
    glfwSetCursorPosCallback(w, register_mouse_movement);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwMakeContextCurrent(w);

    // Cuda stuff
    surface_raii vis(Width, Height);

    // 3D Stuff
    world_geometry world("test_camera_light.obj");

    float spec[3] = {0.2f, 0.4f, 0.2f};
    float diff[3] = {0.1f, 0.9f, 0.7f};
    world.add_light(phong_light(spec, diff), {-1.7f, -1.5f, -1.5f});
    world.add_light(phong_light(spec, diff), {1.3f, -1.8f, -1.2f});
    world.add_light(phong_light(spec, diff), {-1.1f, 2.0f, 1.1f});
    world.add_light(phong_light(spec, diff), {-1.5f, -1.5f, 1.5f});

    const auto& triangles = world.triangles();

    while (!glfwWindowShouldClose(w)) {
        raytrace_many_shaded(vis.getSurface(), world.handle());
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
