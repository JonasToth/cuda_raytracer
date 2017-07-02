#include "visualization.h"
#include <stdexcept>


namespace {
void quit_with_q(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_Q && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
}

visualization::visualization(int width, int height)
    : __width(width)
    , __height(height)
{
    auto init_value = glfwInit();

    if(init_value == 0)
        throw std::runtime_error{"Could not initialize glfw"};

    __window = glfwCreateWindow(__width, __height, "Cuda Raytracer", nullptr, nullptr);

    if(__window == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error{"Could not create window"};
    }

    glfwSetKeyCallback(__window, quit_with_q);
    glfwMakeContextCurrent(__window);

    __initialize_texture();
    __initialize_cuda_surface();

    while(!glfwWindowShouldClose(__window)) {
        glfwSwapBuffers(__window);
        glfwPollEvents();
    }
}

visualization::~visualization() 
{
    // Destroy all cuda and opengl connections
    cudaGraphicsUnmapResources(1, &__cuda_resource);
    cudaDestroySurfaceObject(__cuda_surface);

    // Destroy the window and opengl context
    glfwDestroyWindow(__window);
    glfwTerminate();
}

bool visualization::looping() 
{
    __render_gl_texture();

    glfwSwapBuffers(__window);
    glfwPollEvents();

    return !glfwWindowShouldClose(__window);
}


// https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda
void visualization::__initialize_texture() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &__texture);

    if(__texture == 0)
        throw std::runtime_error{"Could not create opengl texture"};

    glBindTexture(GL_TEXTURE_2D, __texture);
    { // beauty stuff for opengl, maybe skip?
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, __width, __height, 0, GL_RGBA, 
					 GL_UNSIGNED_BYTE, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    const auto E = cudaGraphicsGLRegisterImage(&__cuda_resource, __texture, GL_TEXTURE_2D, 
                                               cudaGraphicsRegisterFlagsWriteDiscard);

    // error checking on the cuda call
    switch (E) {
        case cudaErrorInvalidDevice: throw std::runtime_error{"Cuda bind texture: invalid device"};
        case cudaErrorInvalidValue: throw std::runtime_error{"Cuda bind texture: invalid value"};
        case cudaErrorInvalidResourceHandle: throw std::runtime_error{"Cuda bind texture: invalid resource handle"};
        case cudaErrorUnknown: throw std::runtime_error{"Cuda bind texture: unknown error"};
        default: break;
    }

    // Memory mapping
    cudaGraphicsMapResources(1, &__cuda_resource); 
}

void visualization::__initialize_cuda_surface()
{
    // source: Internet :)
    // https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda
    cudaArray_t __cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&__cuda_array, __cuda_resource, 0, 0);

    cudaResourceDesc cuda_array_resource_desc;
    cuda_array_resource_desc.resType = cudaResourceTypeArray;
    cuda_array_resource_desc.res.array.array = __cuda_array;

    // Surface creation
    cudaCreateSurfaceObject(&__cuda_surface, &cuda_array_resource_desc); 
}


void visualization::__render_gl_texture()
{
    glBindTexture(GL_TEXTURE_2D, __texture);
    {
        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f); 
        }
        glEnd();
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
}
