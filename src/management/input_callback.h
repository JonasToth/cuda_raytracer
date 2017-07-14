#ifndef INPUT_CALLBACK_H_TIZFXPK4
#define INPUT_CALLBACK_H_TIZFXPK4

/// fwd declare, to reduce memory IO
struct GLFWwindow;

/// Callback for glfw to handle key events, that are simply registered in input_manager
void register_key_press(GLFWwindow* w, int key, int scancode, int action, int mods);

/// Callback for glfw to handle mouse movement, that are simply registered in
/// input_manager
void register_mouse_movement(GLFWwindow* w, double xpos, double ypos);


#endif /* end of include guard: INPUT_CALLBACK_H_TIZFXPK4 */
