integration_render::integration_render(std::string obj_name, std::string img_name)
  : obj_name(obj_name)
  , img_name(img_name)
#ifdef __CUDACC__
  , w(width, height, obj_name + " Scene")
#endif
  , render_surface(width, height)
  , scene(obj_name + ".obj")
{
    std::clog << "Setup Rendering Platform initialized" << std::endl;
}


void integration_render::init_default()
{
    // Light Setup similar to blender (position and stuff taken from there)
    float spec[3] = {0.8f, 0.8f, 0.8f};
    float diff[3] = {0.8f, 0.8f, 0.8f};
    scene.add_light(phong_light(spec, diff), {-1.7f, -1.5f, -1.5f});
    scene.add_light(phong_light(spec, diff), {1.3f, -1.8f, -1.2f});
    scene.add_light(phong_light(spec, diff), {-1.1f, 2.0f, 1.1f});
    scene.add_light(phong_light(spec, diff), {-1.5f, -1.5f, 1.5f});

    scene.set_camera(camera(width, height, {-1.5f, 1.2f, -1.5f}, {1.7f, -1.4f, 1.7f}));

    std::clog << "World initialized" << std::endl;
}

void integration_render::run()
{
#ifndef __CUDACC__ // Ray Memory as surface
    render_flat(render_surface, scene.handle());
#else // OpenGL and cuda stuff as surface
    render_flat(render_surface.getSurface(), scene.handle());
    std::this_thread::sleep_for(std::chrono::seconds(delay));
    render_surface.render_gl_texture();
    glfwSwapBuffers(w.getWindow());
#endif
    std::clog << "World rendered" << std::endl;

    render_surface.save_as_png(img_name + ".png");
}
