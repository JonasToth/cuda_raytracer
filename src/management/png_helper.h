#ifndef PNG_HELPER_H_7QDZO3IN
#define PNG_HELPER_H_7QDZO3IN

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <png-plusplus/image.hpp>
#include <png-plusplus/rgba_pixel.hpp>

inline png::image<png::rgba_pixel> memory_to_png(const std::vector<uint8_t>& memory,
                                                 std::size_t width, std::size_t height,
                                                 const int channels)
{
    png::image<png::rgba_pixel> img(width, height);
    for (std::size_t y = 0ul; y < height; ++y) {
        for (std::size_t x = 0ul; x < width; ++x) {
            const auto idx = channels * (y * width + x);
            const png::rgba_pixel pixel(memory[idx], memory[idx + 1], memory[idx + 2]);
            // Otherwise its upside down, because opengl
            img.set_pixel(x, height - y - 1, pixel);
        }
    }
    return img;
}

#pragma GCC diagnostic pop

#endif /* end of include guard: PNG_HELPER_H_7QDZO3IN */
