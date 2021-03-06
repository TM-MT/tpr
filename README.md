# PCR-like Tree Partitioning Reduction

A tridiagonal system solver for CPU and GPU.

[![CMake](https://github.com/TM-MT/tpr/actions/workflows/cmake.yml/badge.svg)](https://github.com/TM-MT/tpr/actions/workflows/cmake.yml)

## Requirements
 - cmake >= 3.8.0
 - g++ >= 9.0.0
 - PAPI == papi-5-7-0-t, if using HWPC

#### Requirements for GPU with CUDA
 - cuda >= 11.0

## Clone & Build & Run

```sh
$ git clone --recurse-submodules https://github.com/TM-MT/tpr.git
$ cd tpr
$ mkdir build && cd build

# CPU Codes
$ export CC=gcc CXX=g++
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make
# run sample code
$ ./src/tpr_main
# run benchmark program
# ./src/tpr_pm N S iter_time Solver
$ ./src/tpr_pm 2048 512 1000 PTPR
# Evaluate the solution by L2-Norm defined by `||x - x_t||_2 / ||x_t||_2` where x_t denotes the truth
# print the result to stdout in csv format,`InputMatrix,Solver,N,S,L2-Norm`
$ ./src/nstab

# Use cuda (cmake < 3.18)
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-gencode arch=compute_75,code=sm_75"  -Dwith_ACC=no -DRandom_BuildTests=no -DBUILD_CUDA=yes ..
# Use cuda (cmake >= 3.20)
$ cmake -DCMAKE_BUILD_TYPE=Release -DCU_ARCH="60" -Dwith_ACC=no -DRandom_BuildTests=no -DBUILD_CUDA=yes ..
```

#### CMake Options
 - `-D CMAKE_BUILD_TYPE={Release|Debug}`: Build type
 - `-D with_PAPI={path/to/papi|OFF}` : Specify path to PAPI installed directory. The default is `OFF`
 - `-D REAL_TYPE={float|double}`: Specify real type of floating point. The default is `float`.
 - `-D TPR_PERF={no|yes}`: Enable Performance monitoring for each stage in TPR. This option may affect the performance.
 - `-D with_LAPACK={no|yes}`: Enable REFERENCE LAPACK `sgtsv` function. Require [Reference LAPACK](http://netlib.org/lapack/) and LAPACKE, and can be found by `FindLAPACK` module provided by cmake.
 - `-D ENABLE_TEST={no|yes}`: Enable building test. The default is `no`
 - `-D BUILD_CUDA={no|yes}`: Build programs written in CUDA. The default is `no`
 - `-D CU_ARCH={CMAKE_CUDA_ARCHITECTURES}`: (cmake >= 3.18) CUDA architectures. ex) ITO-B(Tesla P100) -> "60"

## Documents
use Doxygen

## References
 - Adri??n P. Di??guez, Margarita Amor, and Ram??n Doallo. 2019. Tree Partitioning Reduction: A New Parallel Partition Method for Solving Tridiagonal Systems. ACM Trans. Math. Softw. 45, 3, Article 31 (August 2019), 26 pages. DOI:[https://doi.org/10.1145/3328731](https://doi.org/10.1145/3328731)

