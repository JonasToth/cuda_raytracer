#ifndef MEMORY_SURFACE_H_ARRMLDCK
#define MEMORY_SURFACE_H_ARRMLDCK

#include <cstdint>
#include <string>
#include <vector>

struct pixel_rgba {
    uint8_t r, g, b, a;
};

class memory_surface
{
public:
    memory_surface(std::size_t width, std::size_t height);

    int channel() const noexcept { return __channels; }
    std::size_t width() const noexcept { return __width; }
    std::size_t height() const noexcept { return __height; }

    void write_pixel(std::size_t x, std::size_t y, pixel_rgba p);
    void save_as_png(const std::string& file_name) const;

    const uint8_t* data() const noexcept { return __buffer.data(); }
    uint8_t* data() noexcept { return __buffer.data(); }

private:
    const int __channels = 4;
    const std::size_t __width;
    const std::size_t __height;
    std::vector<uint8_t> __buffer;
};

#endif /* end of include guard: MEMORY_SURFACE_H_ARRMLDCK */
