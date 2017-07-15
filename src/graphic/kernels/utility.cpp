#include <algorithm>
#include "graphic/kernels/utility.h"


void black_kernel(memory_surface& surface)
{
    const auto w = surface.width();
    const auto h = surface.height();
    const auto c = surface.channel();
    std::fill(surface.data(), surface.data() + (w * h * c), 0);
}

void stupid_colors(memory_surface& surface, float t)
{
    for(std::size_t y = 0; y < surface.height(); ++y)
    {
        for(std::size_t x = 0; x < surface.width(); ++x)
        {
            pixel_rgba color;
            uint8_t new_t = t;
            color.r = x - new_t;
            color.g = y + new_t;
            color.b = new_t;
            color.a = 255;
            surface.write_pixel(x, y, color);
        }
    }
}
