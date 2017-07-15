#ifndef MEMORY_SURFACE_H_ARRMLDCK
#define MEMORY_SURFACE_H_ARRMLDCK

#include <cstdint>
#include <string>
#include <vector>

class memory_surface
{
public:
    memory_surface(std::size_t width, std::size_t height);

    void save_as_png(const std::string& file_name) const;
private:
    const int __channels = 4;
    const std::size_t __width;
    const std::size_t __height;
    std::vector<uint8_t> __buffer;
};

#endif /* end of include guard: MEMORY_SURFACE_H_ARRMLDCK */
