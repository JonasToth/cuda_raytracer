#include "management/memory_surface.h"
#include "management/png_helper.h"


memory_surface::memory_surface(std::size_t width, std::size_t height)
  : __width{width}
  , __height{height}
  , __buffer(width * height * __channels)
{
}

void memory_surface::save_as_png(const std::string& file_name) const
{
    // const not allowed, IDK why
    auto img = memory_to_png(__buffer, __width, __height, __channels);
    img.write(file_name);
}
