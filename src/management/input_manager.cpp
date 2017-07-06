#include "input_manager.h"


input_manager& input_manager::instance() {
    static input_manager instance;
    return instance;
}

bool input_manager::isPressed(int key_id) const
{
    auto v = __key_mapping.find(key_id);
    if(v == __key_mapping.end()) 
        return false;
    else 
        return v->second;
}

void input_manager::move_mouse(double new_x, double new_y) noexcept
{
    __x_diff = new_x - __x_pos;
    __y_diff = new_y - __y_pos;

    __x_pos = new_x;
    __y_pos = new_y;
}

void input_manager::clear()
{
    __key_mapping.clear();
    __x_pos = 0.;
    __y_pos = 0.;
    __x_diff = 0.;
    __y_diff = 0.;
}
