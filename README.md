# Basic Raytracer in CUDA

*CUDA Raytracer* is a University project to implement a basic Raytracer in CUDA.
It is supplementary to the lecture on the architecture of parallel computers and gpus from Prof. Froitzheim at the TU Bergakademie Freiberg.

## Defined Goal

The defined scope is to raytrace a simple .obj-file that consists only of triangles and allow basic shading.
More advanced features might be added later.

- import .obj-file, containing only triangles
- implement functional and efficient data structures for the world geometry
- real time output onto screen and/or output to image files
- implementation with modern C++14 and CUDA 8
- Phong Shading
- moveable light sources
- camera movement with user input

Parts of the implementation might rely on previous projects with similar goals developed by former students.

## Platform

This project is developed and tested on the following platform. Other platforms might work but are not tested.

- Ryzen R7 1700X
- GTX 1070
- ubuntu 17.04
- CUDA 8
- clang-4 (/ gcc 5.4)

## Dependencies

Dependencies exist to common libraries and SDKs.

- CUDA 8 SDK
- google test `sudo apt-get install googletest # ubuntu`
