
inline void raytrace_many_shaded(cudaSurfaceObject_t surface, 
                                 world_geometry::data_handle dh)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid((dh.cam.width() + dimBlock.x) / dimBlock.x,
                 (dh.cam.height() + dimBlock.y) / dimBlock.y);
    black_kernel<<<dimGrid, dimBlock>>>(surface, dh.cam.width(), dh.cam.height());
    trace_triangles_shaded<<<dimGrid, dimBlock>>>(surface, dh.cam,
                                                  dh.triangles, dh.triangle_count, 
                                                  dh.lights, dh.light_count,
                                                  dh.cam.width(), dh.cam.height());
}


