
inline void raytrace_many_cuda(cudaSurfaceObject_t Surface, const camera& c,
                               const triangle* Triangles, int TriangleCount)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((c.width() + dimBlock.x) / dimBlock.x,
                 (c.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(Surface, c.width(), c.height());
    trace_many_triangles_with_camera<<<dimGrid, dimBlock>>>(
        Surface, c, Triangles, TriangleCount, c.width(), c.height());
}


inline void raytrace_cuda(cudaSurfaceObject_t& Surface, int width, int height,
                          const triangle* T)
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x) / dimBlock.x, (height + dimBlock.y) / dimBlock.y);
    trace_single_triangle<<<dimGrid, dimBlock>>>(Surface, T, width, height);
}
