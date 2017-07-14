#ifndef INTEGRATION_RENDER_H_QHF9TGEZ
#define INTEGRATION_RENDER_H_QHF9TGEZ

#include <chrono>
#include <thread>

#include "management/window.h"
#include "management/world.h"
#include "util/kernel_launcher/world_shading.h"

/// Define the basic sekeleton for an integration test, that renders an image,
/// that will be compared with an reference image.
/// 
/// Will be used in conjunction with ./validate.sh


class integration_render
{
public:
    integration_render(std::string name);

    // default scene, that is used by everybody seemingly
    void init_default();

    // render the scene and write output file
    void run();

    // give possibility to adjust the scene if necessary
    world_geometry& getScene() noexcept { return scene; }

private:
    const std::string test_name;
    const std::string out_prefix = "./int_test_output/";
    const std::string in_prefix  = "./int_test_input/";

    window w;
    surface_raii render_surface;
    camera c;
    world_geometry scene;
};


#include "integration_render.inl"


#endif /* end of include guard: INTEGRATION_RENDER_H_QHF9TGEZ */
