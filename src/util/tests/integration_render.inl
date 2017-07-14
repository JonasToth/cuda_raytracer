integration_render::integration_render(std::string name)
    : test_name(name)
    , w(800, 600, test_name + " Scene")
    , render_surface(w.getWidth(), w.getHeight())
    , c(w.getWidth(), w.getHeight(), {-1.5f, 1.2f, -1.5f}, {1.7f, -1.4f, 1.7f})
    , scene(test_name + ".obj")
{
    auto w_ptr = w.getWindow();
    glfwMakeContextCurrent(w_ptr);
    std::clog << "Setup Rendering Platform initialized" << std::endl;
}


void integration_render::init_default()
{
    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {-1.7f, -1.5f, -1.5f});
    scene.add_light(phong_light(spec, diff), { 1.3f, -1.8f, -1.2f});
    scene.add_light(phong_light(spec, diff), {-1.1f,  2.0f,  1.1f});
    scene.add_light(phong_light(spec, diff), {-1.5f, -1.5f,  1.5f});

    std::clog << "World initialized" << std::endl;
}

void integration_render::run()
{
    const auto& triangles = scene.triangles();
    raytrace_many_shaded(render_surface.getSurface(), c,
                         triangles.data().get(), triangles.size(),
                         scene.lights().data().get(), scene.light_count());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    render_surface.render_gl_texture();
    render_surface.save_as_png(test_name + ".png");
    std::clog << "World rendered" << std::endl;
}
