#ifndef IMAGE_LOOP_MACRO_H_MPZV3DZG
#define IMAGE_LOOP_MACRO_H_MPZV3DZG

#ifndef PIXEL_LOOP
#define PIXEL_LOOP(surface)                                \
    for (std::size_t y = 0ul; y < (surface).height(); ++y) \
        for (std::size_t x = 0ul; x < (surface).width(); ++x)

#endif

#endif /* end of include guard: IMAGE_LOOP_MACRO_H_MPZV3DZG */
