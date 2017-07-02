#define TINYOBJLOADER_IMPLEMENTATION
#include "obj_io.h"

#include <iostream>
#include <stdexcept>

void WorldGeometry::load(const std::string& file_name) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file_name.c_str());

    if(!err.empty()) {
        std::cerr << err << std::endl;
    }

    if(ret == false) {
        std::string error_message = "Could not read the file ";
        throw std::invalid_argument{error_message + file_name};
    }
}
