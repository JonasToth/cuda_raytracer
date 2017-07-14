/// Surface gets all black
__global__ void black_kernel(cudaSurfaceObject_t surface, int width, int height)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 black;
    black.x = 0;
    black.y = 0;
    black.z = 0;
    black.w = 255;

    if (x < width && y < height)
        surf2Dwrite(black, surface, x * sizeof(black), y);
}

/// Just draw some color on the surface, control with parameter t
__global__ void stupid_colors(cudaSurfaceObject_t surface, int width, int height, float t)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 color;
        char new_t = t;
        color.x = x - new_t;
        color.y = y + new_t;
        color.z = new_t;
        color.w = 255;
        surf2Dwrite(color, surface, x * sizeof(color), y);
    }
}
