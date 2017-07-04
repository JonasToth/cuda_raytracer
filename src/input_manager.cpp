#include "input_manager.h"


input_manager& input_manager::instance() {
    static input_manager instance;
    return instance;
}

bool input_manager::isPressed(int key_id) 
{
    auto v = __key_mapping.find(key_id);
    if(v == __key_mapping.end()) 
        return false;
    else 
        return v->second;
}
