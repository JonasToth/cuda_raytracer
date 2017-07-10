#ifndef VISUALIZATION_H_8LNOECHQ
#define VISUALIZATION_H_8LNOECHQ

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <string>

/// Controls the output to the screen
class surface_raii {
public:
    surface_raii(int width, int height);
    ~surface_raii();

    cudaSurfaceObject_t& getSurface() noexcept { return __cuda_surface; }
    void render_gl_texture() noexcept;

    void save_as_png(const std::string& file_name) const;

private:
    void __initialize_texture();
    void __initialize_cuda_surface();

    int __width;
    int __height;

    GLuint __texture;

    cudaArray_t __cuda_array;
    cudaResourceDesc __cuda_array_resource_desc;
    cudaGraphicsResource_t __cuda_resource;

    cudaSurfaceObject_t __cuda_surface;
};


#endif /* end of include guard: VISUALIZATION_H_8LNOECHQ */
