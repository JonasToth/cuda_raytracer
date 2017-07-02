#include "gtest/gtest.h"
#include "macros.h"
#include "triangle.h"
#include "ray.h"
#include "surface_raii.h"

#include <GLFW/glfw3.h>
#include <gsl/gsl>
#include <iostream>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <utility>


static void quit_with_q(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}


__global__ void grayKernel(cudaSurfaceObject_t Surface, int width, int height, float t)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        uchar4 Color;
        char new_t = t;
        Color.x = x - new_t;
        Color.y = y + new_t;
        Color.z = new_t;
        Color.w = 255;
        surf2Dwrite(Color, Surface, x * 4, y);
    }
}

void invokeRenderingKernel(cudaSurfaceObject_t& Surface, float t)
{
    //std::clog << "Rendering new image " << char{t} << std::endl;
    dim3 dimBlock(32,32);
    dim3 dimGrid((640 + dimBlock.x) / dimBlock.x,
                 (480 + dimBlock.y) / dimBlock.y);
    std::clog << "Render : " << t << std::endl;
    grayKernel<<<dimGrid, dimBlock>>>(Surface, 640, 480, t);
}

TEST(cuda_draw, basic_drawing) {
    auto Initialized = glfwInit();
    ASSERT_NE(Initialized, 0) << "Could not init glfw";

    gsl::owner<GLFWwindow*> w = glfwCreateWindow(640, 480, "Cuda Raytracer", nullptr, nullptr);
    glfwSetKeyCallback(w, quit_with_q);
    glfwMakeContextCurrent(w);

    surface_raii vis(640, 480);

    std::clog << "Init" << std::endl;
    float t = 0.f;
    while(!glfwWindowShouldClose(w)) {
        std::clog << "Loop" << std::endl;
        t += 0.1f;
        invokeRenderingKernel(vis.getSurface(), t);

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwPollEvents();
        std::clog << "Loop end" << std::endl;
    }

    std::clog << "Done" << std::endl;
    glfwDestroyWindow(w);
    glfwTerminate();
}

/// Write pixel data with cuda.
void render_cuda2(cudaSurfaceObject_t& Surface, float t) {
    // Rendering
    invokeRenderingKernel(Surface, t);
}

TEST(cuda_draw, drawing_less_surfaces) {
    auto Initialized = glfwInit();
    ASSERT_NE(Initialized, 0) << "Could not init glfw";

    gsl::owner<GLFWwindow*> w = glfwCreateWindow(640, 480, "Cuda Raytracer", nullptr, nullptr);
    glfwSetKeyCallback(w, quit_with_q);
    glfwMakeContextCurrent(w);

    surface_raii vis(640, 480);

    float t = 0.f;
    while(!glfwWindowShouldClose(w)) {
        t += 0.1f;
        render_cuda2(vis.getSurface(), t);

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    std::clog << "Done" << std::endl;
    glfwDestroyWindow(w);
    glfwTerminate();
}

__global__ void black_kernel(cudaSurfaceObject_t Surface, int Width, int Height) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 BGColor;
    BGColor.x = 0;
    BGColor.y = 0;
    BGColor.z = 0;
    BGColor.w = 255;

    if(x < Width && y < Height)
        surf2Dwrite(BGColor, Surface, x * 4, y);
}

__global__ void trace_kernel(cudaSurfaceObject_t Surface, triangle* T, int Width, int Height) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const float focal_length = .5f;

    if(x < Width && y < Height)
    {
        ray R;
        R.origin    = coord{0., 0., 2.};
        float DX = 2.f / ((float) Width  - 1);
        float DY = 2.f / ((float) Height - 1);
        R.direction = coord{x * DX - 1.f, y * DY - 1.f, focal_length};

        uchar4 FGColor;
        FGColor.x = 255;
        FGColor.y = 255;
        FGColor.z = 255;
        FGColor.w = 255;

        
        const auto Traced = R.intersects(*T);

        if(Traced.first) {
            surf2Dwrite(FGColor, Surface, x * 4, y);
        }
        //else {
            //surf2Dwrite(BGColor, Surface, x * 4, y);
        //}
    }
}

void raytrace_cuda(cudaSurfaceObject_t& Surface, triangle* T) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((640 + dimBlock.x) / dimBlock.x,
                 (480 + dimBlock.y) / dimBlock.y);
    trace_kernel<<<dimGrid, dimBlock>>>(Surface, T, 640, 480);
}

TEST(cuda_draw, drawing_traced_triangle) 
{
    auto Initialized = glfwInit();
    ASSERT_NE(Initialized, 0) << "Could not init glfw";

    gsl::owner<GLFWwindow*> w = glfwCreateWindow(640, 480, "Cuda Raytracer", nullptr, nullptr);
    glfwSetKeyCallback(w, quit_with_q);
    glfwMakeContextCurrent(w);

    std::clog << "before surface creation" << std::endl;

    surface_raii vis(640, 480);
    
    std::clog << "init" << std::endl;

    // Create the Triangle and Coordinates on the device
    thrust::device_vector<coord> Vertices(5);
    //Vertices[0] = {.5f,-1,1}; 
    //Vertices[1] = {-1,.5f,1};
    //Vertices[2] = {1,1,1};
    Vertices[0] = {0,-1,1}; 
    Vertices[1] = {-1,1,1};
    Vertices[2] = {1,1,1};
    Vertices[3] = {1,-0.8,1};
    Vertices[4] = {-1,0.8,1};

    const thrust::device_ptr<coord> P0 = &Vertices[0];
    const thrust::device_ptr<coord> P1 = &Vertices[1];
    const thrust::device_ptr<coord> P2 = &Vertices[2];
    const thrust::device_ptr<coord> P3 = &Vertices[3];
    const thrust::device_ptr<coord> P4 = &Vertices[4];

    thrust::device_vector<triangle> Triangles(3);
    Triangles[0] = {P0.get(), P1.get(), P2.get()};
    Triangles[1] = {P0.get(), P1.get(), P3.get()};
    Triangles[2] = {P4.get(), P2.get(), P0.get()};
    std::clog << "triangles done" << std::endl;

    while(!glfwWindowShouldClose(w)) {
        dim3 dimBlock(32,32);
        dim3 dimGrid((640 + dimBlock.x) / dimBlock.x,
                     (480 + dimBlock.y) / dimBlock.y);
        black_kernel<<<dimGrid, dimBlock>>>(vis.getSurface(), 640, 480);

        std::clog << "loop" << std::endl;
        for(std::size_t i = 0; i < Triangles.size(); ++i)
        {
            std::clog << "trace" << std::endl;
            const thrust::device_ptr<triangle> T = &Triangles[i];
            raytrace_cuda(vis.getSurface(), T.get());
        }

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwWaitEvents();
    } 
    std::clog << "Done" << std::endl;
    glfwDestroyWindow(w);
    glfwTerminate();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
