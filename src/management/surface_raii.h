#ifndef VISUALIZATION_H_8LNOECHQ
#define VISUALIZATION_H_8LNOECHQ

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

enum class render_target {
    texture, ///< render to an opengl texture, visualizable
    memory,  ///< render to plain memory, just output to image
};

/// Controls the output to the screen
class surface_raii
{
public:
    surface_raii(int width, int height, render_target target = render_target::texture);
    ~surface_raii();

    cudaSurfaceObject_t& getSurface() noexcept { return __cuda_surface; }
    void render_gl_texture() noexcept;

    void save_as_png(const std::string& file_name) const;

private:
    /// Decide if opengl or raw memory
    void __initialize_render_target();
    /// Handle opengl
    void __initialize_opengl_texture();
    /// Handle raw memory
    void __initialize_memory_texture();
    /// Create CudaSurface from opengl
    void __initialize_cuda_surface();

    /// Helper to extract memory from which resource ever
    std::vector<uint8_t> __get_texture_memory() const;

    const int __channels = 4;    ///< RGBA constant
    enum render_target __target; ///< flag for opengl or memory
    int __width;                 ///< width of the texture
    int __height;                ///< height of the texture

    std::vector<uint8_t> __memory_texture; ///< when rendering to plain memory, buffer
    GLuint __texture; ///< when using an opengl texture, this value will be set

    /// When using an gpu, this will be used to handle the memory there
    cudaArray_t __cuda_array;
    cudaResourceDesc __cuda_array_resource_desc;
    cudaGraphicsResource_t __cuda_resource;

    cudaSurfaceObject_t __cuda_surface;
};


#endif /* end of include guard: VISUALIZATION_H_8LNOECHQ */
