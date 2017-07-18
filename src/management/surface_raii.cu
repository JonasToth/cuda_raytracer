#include "surface_raii.h"

#include <gsl/gsl>
#include <stdexcept>
#include "management/png_helper.h"

surface_raii::surface_raii(int width, int height)
  : __width{width}
  , __height{height}
  , __texture{0}
{
    __initialize_opengl_texture();
    __initialize_cuda_surface();
}

surface_raii::~surface_raii()
{
    // Destroy the opengl texture
    glDeleteTextures(1, &__texture);
    glFinish();
    // Destroy link between cuda and opengl
    cudaGraphicsUnmapResources(1, &__cuda_resource);

    cudaDestroySurfaceObject(__cuda_surface);
}

void surface_raii::save_as_png(const std::string& file_name) const
{
    const auto memory = __get_texture_memory();
    // const not allowed, IDK why
    auto img = memory_to_png(memory, __width, __height, __channels);
    img.write(file_name);
}


void surface_raii::__initialize_opengl_texture()
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &__texture);

    if (__texture == 0)
        throw std::runtime_error{"Could not create opengl texture"};

    glBindTexture(GL_TEXTURE_2D, __texture);
    { // beauty stuff for opengl, maybe skip?
        Expects(__channels == 4);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, __width, __height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();

    const auto E = cudaGraphicsGLRegisterImage(&__cuda_resource, __texture, GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsWriteDiscard);

    // error checking on the cuda call
    switch (E) {
        case cudaErrorInvalidDevice:
            throw std::runtime_error{"Cuda bind texture: invalid device"};
        case cudaErrorInvalidValue:
            throw std::runtime_error{"Cuda bind texture: invalid value"};
        case cudaErrorInvalidResourceHandle:
            throw std::runtime_error{"Cuda bind texture: invalid resource handle"};
        case cudaErrorUnknown:
            throw std::runtime_error{"Cuda bind texture: unknown error"};
        default:
            break;
    }

    // Memory mapping
    cudaGraphicsMapResources(1, &__cuda_resource);
}

void surface_raii::__initialize_cuda_surface()
{
    // source: Internet :)
    // https://stackoverflow.com/questions/19244191/
    // cuda-opengl-interop-draw-to-opengl-texture-with-cuda
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
    cudaGraphicsSubResourceGetMappedArray(&__cuda_array, __cuda_resource, 0, 0);
    __cuda_array_resource_desc.resType = cudaResourceTypeArray;
    __cuda_array_resource_desc.res.array.array = __cuda_array;

    // Surface creation
    cudaCreateSurfaceObject(&__cuda_surface, &__cuda_array_resource_desc);
}

void surface_raii::render_gl_texture() noexcept
{
    glBindTexture(GL_TEXTURE_2D, __texture);
    {
        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(+1.0f, +1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(-1.0f, +1.0f);
        }
        glEnd();
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
}

std::vector<uint8_t> surface_raii::__get_texture_memory() const
{
    std::vector<uint8_t> gl_texture_data(__width * __height * __channels);
    glReadPixels(0, 0, __width, __height, GL_RGBA, GL_UNSIGNED_BYTE,
                 gl_texture_data.data());
    glFinish();

    return gl_texture_data;
}

