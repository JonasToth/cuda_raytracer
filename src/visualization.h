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

private:
    int __width;
    int __height;

    gsl::owner<GLFWwindow*> __window;
};


#endif /* end of include guard: VISUALIZATION_H_8LNOECHQ */
