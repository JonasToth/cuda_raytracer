#include "obj_io.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <gsl/gsl>
#include <iostream>
#include <stdexcept>


WorldGeometry::WorldGeometry() = default;

// Wrap the tiny_obj library
namespace {
    struct loaded_data { 
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
    };

    loaded_data __load(const char* file_name) {
        loaded_data result;
        std::string err;
        bool ret = tinyobj::LoadObj(&result.attrib, &result.shapes, 
                                    &result.materials, &err, file_name);

        if(!err.empty()) {
            std::cerr << err << std::endl;
        }

        if(ret == false) {
            std::string error_message = "Could not read the file ";
            throw std::invalid_argument{error_message + file_name};
        }
        Ensures(result.attrib.vertices.size() % 3 == 0);
        return result;
    }
}

void WorldGeometry::load(const std::string& file_name) {
    // Load with the library
    const auto data = __load(file_name.c_str());
    const auto& v = data.attrib.vertices;

    Expects(v.size() % 3 == 0);

    // transform the data into local represenation
    __vertices.reserve(v.size() / 3); // per vertex 3 floats
    
    for(std::size_t i = 0; i < data.attrib.vertices.size(); i+= 3)
    {
        __vertices.push_back({v[i], v[i+1], v[i+2]});
    }
}
