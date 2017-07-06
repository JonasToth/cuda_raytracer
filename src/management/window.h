#ifndef WINDOW_H_VVLXZ4IZ
#define WINDOW_H_VVLXZ4IZ


#include <gsl/gsl>
#include <GLFW/glfw3.h>
#include <string>


class window {
public:
    window(int width, int height, const std::string& title);
    ~window();

    GLFWwindow* getWindow() noexcept { return __w; }

    int getWidth() const noexcept { return __width; }
    int getHeight() const noexcept { return __height; }

private:
    gsl::owner<GLFWwindow*> __w;

    int __width;
    int __height;
};

#endif /* end of include guard: WINDOW_H_VVLXZ4IZ */
