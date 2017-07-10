#include "surface_raii.h"
#include "CImg.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>


surface_raii::surface_raii(int width, int height)
    : __width{width}
    , __height{height}
    , __texture{0}
    , __cuda_array{}
    , __cuda_array_resource_desc{}
    , __cuda_resource{}
    , __cuda_surface{}
{
    __initialize_texture();
    __initialize_cuda_surface();
}

surface_raii::~surface_raii() 
{
    std::clog << "Destroying the surface and texture" << std::endl;
    // Destroy the opengl texture
    glDeleteTextures(1, &__texture);

    // Destroy all cuda and opengl connections
    cudaDestroySurfaceObject(__cuda_surface);
    cudaGraphicsUnmapResources(1, &__cuda_resource);

}

namespace {
    /// Get correct memory ordering for 8 bit, 3 channel buffer for pixels, buffers preallocated
    void untangle_texture(std::size_t width, std::size_t height, 
                          const uint8_t* tangled, uint8_t* untangled)
    {
        const auto n_pixels = width * height;
        const auto channels = 3;

        for(std::size_t i = 0ul; i < n_pixels; ++i)
        {
            const auto interleaved_index = channels * i;
            untangled[i]                = tangled[interleaved_index];
            untangled[n_pixels + i]     = tangled[interleaved_index + 1];
            untangled[2 * n_pixels + i] = tangled[interleaved_index + 2];
        }
    }
}

/// https://stackoverflow.com/questions/6846762/glreadpixels-image-looks-dithered
void surface_raii::save_as_png(const std::string& file_name) const
{
    // 3 channels
    std::vector<uint8_t> gl_texture_data(__width * __height * 3);
    glReadPixels(0, 0, __width, __height, GL_RGB, GL_UNSIGNED_BYTE, gl_texture_data.data());

    // get memory ordering for CImg
    std::vector<uint8_t> cimg_texture_data(__width * __height * 3);
    untangle_texture(__width, __height, gl_texture_data.data(), cimg_texture_data.data());

    using cimg_library::CImg;
    using cimg_library::CImgDisplay;

    // output image file with CImg
    //CImg<uint8_t> image(untangle_texture.data(), __width, __height, 1, 3);

    // test output
    //CImgDisplay disp(image, "Test output");
    //while(!disp.is_closed())
        //disp.wait();
}


// https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda
void surface_raii::__initialize_texture() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &__texture);

    if(__texture == 0)
        throw std::runtime_error{"Could not create opengl texture"};

    glBindTexture(GL_TEXTURE_2D, __texture);
    { // beauty stuff for opengl, maybe skip?
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, __width, __height, 0, GL_RGB, 
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

void surface_raii::__initialize_cuda_surface()
{
    // source: Internet :)
    // https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda
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
