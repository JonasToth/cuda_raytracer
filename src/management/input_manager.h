#ifndef INPUT_MANAGER_H_ODDJW5OV
#define INPUT_MANAGER_H_ODDJW5OV

#include <unordered_map>
#include <GLFW/glfw3.h>

/// Singleton, that manages the input from glfw
class input_manager
{
public:
    input_manager(const input_manager&) = delete;
    input_manager(input_manager&&) = delete;
    input_manager& operator=(const input_manager&) = delete;
    input_manager& operator=(input_manager&&) = delete;

    static input_manager& instance();

    void press(int key_id) { __key_mapping[key_id] = true; }
    void release(int key_id) { __key_mapping[key_id] = false; }

    bool isPressed(int key_id);

    void move_mouse(double new_x, double new_y) noexcept;
    double mouse_x() const noexcept { return __x_pos; }
    double mouse_y() const noexcept { return __y_pos; }

    double mouse_diff_x() const noexcept { return __x_diff; }
    double mouse_diff_y() const noexcept { return __y_diff; }

    void clear();

private:
    input_manager() = default;

    std::unordered_map<int, bool> __key_mapping;
    double __x_pos  = 0.; ///< stores current cursor x position
    double __y_pos  = 0.; ///< stores current cursor y position
    double __x_diff = 0.; ///< stores difference to last cursor x position
    double __y_diff = 0.; ///< stores difference to last cursor y position
};

#endif /* end of include guard: INPUT_MANAGER_H_ODDJW5OV */
