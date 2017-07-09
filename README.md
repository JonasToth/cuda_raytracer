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

## Installation

First, install dependenices! (CMake, Cuda, OpenGL, Google Test for testing)

```
git clone --recursive https://github.com/JonasToth/cuda_raytracer.git
cd cuda_raytracer
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make raytracer.x -j4 # main executable (dummy right now)
make all -j4 # all tests, demos and benchmarks, RECOMMENDED
```

## Platform

This project is developed and tested on the following platform. Other platforms might work but are not tested.

- Ryzen R7 1700X
- GTX 1070
- ubuntu 17.04
- CUDA 8
- clang 4 and gcc 5.4

## Dependencies

Dependencies exist to common libraries and SDKs.

- CUDA 8 SDK  
  https://developer.nvidia.com/cuda-downloads
- https://github.com/google/googletest  
  `sudo apt-get install libgtest-dev googletest # ubuntu`  
  https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/  
  https://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test  
- GLFW http://www.glfw.org/
  `sudo apt-get install libglfw3-dev`
