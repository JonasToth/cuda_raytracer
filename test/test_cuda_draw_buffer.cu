#include "gtest/gtest.h"
#include "macros.h"

// include before, otherwise compile error
#include <GL/glew.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <gsl/gsl>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <vector_types.h>

static void quit_with_q(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_Q && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}


TEST(cuda_draw, window_and_context_creation) {
    auto InitVal = glfwInit();
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    gsl::owner<GLFWwindow*> Window = glfwCreateWindow(640, 480, "Test CUDA drawing", nullptr, nullptr);
    ASSERT_NE(Window, nullptr) << "Window not created";

    OUT << "Close window and test with q" << std::endl;
    
    // window shall be closed when q is pressed
    glfwSetKeyCallback(Window, quit_with_q);

    // opengl context for drawing
    glfwMakeContextCurrent(Window);

    while(!glfwWindowShouldClose(Window)) {
        glfwPollEvents();
    }

    glfwDestroyWindow(Window);
    glfwTerminate();
}

TEST(cuda_draw, basic_drawing) {
    auto InitVal = glfwInit();
    ASSERT_NE(InitVal, 0) << "Could not initialize GLFW";

    auto GlewStatus = glewInit();
    ASSERT_EQ(GlewStatus, GLEW_OK) << "Could not init glew";

    gsl::owner<GLFWwindow*> Window = glfwCreateWindow(640, 480, "Test CUDA drawing", nullptr, nullptr);
    ASSERT_NE(Window, nullptr) << "Window not created";

    OUT << "Close window and test with q" << std::endl;
    
    // window shall be closed when q is pressed
    glfwSetKeyCallback(Window, quit_with_q);

    // opengl context for drawing
    glfwMakeContextCurrent(Window);

    // register a glTexture, that can be filled black ...
    GLuint Texture = 0;
    glGenBuffers(1, &Texture);
    ASSERT_NE(Texture, 0) << "Could not create gl buffer";

    std::size_t BufferSize = 640 * 480 * sizeof(float3);
    glBufferData(GL_ARRAY_BUFFER, BufferSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    auto e = cudaGLRegisterBufferObject(Texture);
    ASSERT_EQ(e, cudaSuccess) << "Could not register buffer object";

    while(!glfwWindowShouldClose(Window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Write the Texture with Cuda
        //thrust::fill(thrust::device, TexturePointer, TexturePointer + 30, 1.f);

        // Render that texture with OpenGL
        // https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda
        glfwSwapBuffers(Window);
        glfwPollEvents();
    }

    // Clean up the cuda memory mapping
    ASSERT_EQ(e, cudaSuccess) << "Could not unmap the resource";

    // Delete opengl texture
    glDeleteBuffers(1, &Texture);

    glfwDestroyWindow(Window);
    glfwTerminate();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
