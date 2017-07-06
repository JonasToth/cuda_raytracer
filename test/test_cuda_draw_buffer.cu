#include "gtest/gtest.h"

#include "graphic/camera.h"
#include "graphic/triangle.h"
#include "graphic/ray.h"
#include "graphic/kernels/utility.h"
#include "macros.h"
#include "management/input_manager.h"
#include "management/input_callback.h"
#include "management/surface_raii.h"
#include "management/window.h"
#include "obj_io.h"

#include <iostream>
//#include <limits>
#include <utility>

const int Width = 800, Height = 800;
camera c(Width, Height, {2.f, 2.f, 2.f}, {0.f, 0.f, 1.f});

double m_x = 0., m_y = 0.;

static void handle_keys(GLFWwindow* w)
{
    const float dP     = 0.5;
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

    std::clog << "X: " << im.mouse_x() << ";Y: " << im.mouse_y() << std::endl;
    std::clog << "Camera Steering At: " << c.steering() << std::endl;
}

/*static void mouse_scrolling(GLFWwindow* w, double xoffset, double yoffset)
{
    double alpha = 5. * yoffset / (2. * M_PI);

    c.swipe(alpha, 0., 0.);

    std::clog << "Xoff: " << xoffset << ";Yoff: " << yoffset << std::endl;
    std::clog << "Camera Steering At: " << c.steering() << std::endl;
}*/

void invokeRenderingKernel(cudaSurfaceObject_t& Surface, float t)
{
    //std::clog << "Rendering new image " << char{t} << std::endl;
    dim3 dimBlock(32,32);
    dim3 dimGrid((Width  + dimBlock.x) / dimBlock.x,
                 (Height + dimBlock.y) / dimBlock.y);
    std::clog << "Render : " << t << std::endl;
    stupid_colors<<<dimGrid, dimBlock>>>(Surface, Width, Height, t);
}

TEST(cuda_draw, basic_drawing) {
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwMakeContextCurrent(w);

    surface_raii vis(Width, Height);

    std::clog << "Init" << std::endl;
    float t = 0.f;
    while(!glfwWindowShouldClose(w)) {
        std::clog << "Loop" << std::endl;
        t += 0.5f;
        invokeRenderingKernel(vis.getSurface(), t);

        vis.render_gl_texture();

        glfwSwapBuffers(w);
        glfwPollEvents();
        handle_keys(w);
        std::clog << "Loop end" << std::endl;
    }
    input_manager::instance().clear();
    std::clog << "Done" << std::endl;
}


__global__ void trace_kernel(cudaSurfaceObject_t Surface, const triangle* T, int Width, int Height) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const float focal_length = 1.f;

    if(x < Width && y < Height)
    {
        ray R;
        R.origin    = coord{0.f, 0.f, -1.f};
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

void raytrace_cuda(cudaSurfaceObject_t& Surface, const triangle* T) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((Width + dimBlock.x) / dimBlock.x,
                 (Height+ dimBlock.y) / dimBlock.y);
    trace_kernel<<<dimGrid, dimBlock>>>(Surface, T, Width, Height);
}

__global__ void trace_many_kernel(cudaSurfaceObject_t Surface, 
                                  camera c,
                                  const triangle* Triangles, int TriangleCount,
                                  int Width, int Height)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < Width && y < Height)
    {
        ray R = c.rayAt(x, y);

        uchar4 FGColor;
        FGColor.x = 255;
        FGColor.y = 255;
        FGColor.z = 255;
        FGColor.w = 255;

        uchar4 BGColor;
        BGColor.x = 0;
        BGColor.y = 0;
        BGColor.z = 0;
        BGColor.w = 255;

        triangle const* NearestTriangle = nullptr;
        intersect NearestIntersect;
        //NearestIntersect.depth = std::numeric_limits<float>::max;
        NearestIntersect.depth = 10000.f;

        // Find out the closes triangle
        for(std::size_t i = 0; i < TriangleCount; ++i)
        {
            const auto Traced = R.intersects(Triangles[i]);
            if(Traced.first)
            {
                if(Traced.second.depth < NearestIntersect.depth)
                {
                    NearestTriangle = &Triangles[i];
                    NearestIntersect = Traced.second;
                }
            }
        }

        if(NearestTriangle != nullptr) {
            auto flat_shade = dot(NearestIntersect.normal, R.direction);
            //FGColor.x = NearestIntersect.depth * 10.f;
            FGColor.y = NearestIntersect.depth * 10.f;
            FGColor.z = NearestIntersect.depth * 10.f;
            FGColor.x = flat_shade;
            //FGColor.y = flat_shade * 255;
            //FGColor.z = flat_shade * 255;
            surf2Dwrite(FGColor, Surface, x * 4, y);
        }
        else {
            surf2Dwrite(BGColor, Surface, x * 4, y);
        }
    }

}

void raytrace_many_cuda(cudaSurfaceObject_t& Surface, 
                        const camera& c,
                        const triangle* Triangles,
                        int TriangleCount) {
    dim3 dimBlock(32,32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    trace_many_kernel<<<dimGrid, dimBlock>>>(Surface, c, Triangles, TriangleCount, 
                                             c.width(), c.height());
}

TEST(cuda_draw, drawing_traced_triangle) 
{
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwMakeContextCurrent(w);

    std::clog << "before surface creation" << std::endl;

    surface_raii vis(Width, Height);
    
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

    const auto P0 = Vertices[0];
    const auto P1 = Vertices[1];
    const auto P2 = Vertices[2];
    const auto P3 = Vertices[3];
    const auto P4 = Vertices[4];

    thrust::device_vector<triangle> Triangles(3);
    Triangles[0] = {P0, P1, P2};
    Triangles[1] = {P0, P1, P3};
    Triangles[2] = {P4, P2, P0};
    std::clog << "triangles done" << std::endl;

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
        glfwWaitEvents();
        handle_keys(w);
    } 
    input_manager::instance().clear();
    std::clog << "Done" << std::endl;
}

TEST(cuda_draw, draw_loaded_geometry)
{
    // Window stuff
    window win(Width, Height, "Cuda Raytracer");
    auto w = win.getWindow();

    glfwSetKeyCallback(w, register_key_press);
    glfwSetCursorPosCallback(w, register_mouse_movement);
    //glfwSetScrollCallback(w, mouse_scrolling);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //c.lookAt({0.f, 0.f, 0.f});
    std::clog << c.steering() << std::endl;

    glfwMakeContextCurrent(w);

    // Cuda stuff
    surface_raii vis(Width, Height);

    // 3D Stuff
    world_geometry world("shapes.obj");
    std::clog << "initialized" << std::endl;

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
    std::clog << "Done" << std::endl;
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
