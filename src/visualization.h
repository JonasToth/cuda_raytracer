#ifndef VISUALIZATION_H_8LNOECHQ
#define VISUALIZATION_H_8LNOECHQ

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <gsl/gsl>

/// Controls the output to the screen
class visualization {
public:
    visualization(int width, int height);
    ~visualization();

    GLFWwindow* getWindow() noexcept { return __window; }
    cudaSurfaceObject_t& getSurface() noexcept { return __cuda_surface; }

    bool looping();


private:
    void __initialize_texture();
    void __initialize_cuda_surface();

    void __render_gl_texture();

    int __width;
    int __height;

    gsl::owner<GLFWwindow*> __window;

    GLuint __texture;
    cudaGraphicsResource_t __cuda_resource;

    cudaSurfaceObject_t __cuda_surface;
};


#endif /* end of include guard: VISUALIZATION_H_8LNOECHQ */
