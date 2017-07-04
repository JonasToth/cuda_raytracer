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

private:
    input_manager() = default;

    std::unordered_map<int, bool> __key_mapping;
};

#endif /* end of include guard: INPUT_MANAGER_H_ODDJW5OV */
