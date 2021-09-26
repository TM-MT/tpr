# Tree Partitioning Reduction

## Requirements
 - cmake >= 3.8.0
 - g++ >= 9.0.0
 - PAPI == papi-5-7-0-t, if using HWPC

#### Requirements for GPU Run
 - nvc++ >= 20.11-0
 - cuda 11.0

## Clone & Build & Run

```sh
$ git clone --recurse-submodules https://github.com/TM-MT/tpr.git
$ cd tpr
$ mkdir build && cd build
# SPECIFY COMPILER
$ export CC=gcc CXX=g++
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make
# run sample code
$ ./src/tpr_main
# run with PMlib Reporting
$ ./src/tpr_pm

# Use GPU with OpenACC
$ export CC=nvc CXX=nvc++
$ cmake -D CMAKE_BUILD_TYPE=Release -Dwith_ACC=yes -DCMAKE_C_FLAGS="-noswitcherror -ta=tesla:managed" -DCMAKE_CXX_FLAGS="-noswitcherror -ta=tesla:managed" -DRandom_BuildTests=off  ..

# Use cuda (cmake < 3.18)
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_FLAGS="-noswitcherror" -DCMAKE_C_FLAGS="-noswitcherror" -DCMAKE_CUDA_FLAGS="-gencode arch=compute_75,code=sm_75"  -Dwith_ACC=no -DRandom_BuildTests=no -DBUILD_CUDA=yes ..
# Use cuda (cmake >= 3.18)
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_FLAGS="-noswitcherror" -DCMAKE_C_FLAGS="-noswitcherror" -DCU_ARCH="60" -Dwith_ACC=no -DRandom_BuildTests=no -DBUILD_CUDA=yes ..
```

#### Options
 - `-D CMAKE_BUILD_TYPE={Release|Debug}`: Build type
 - `-D with_PAPI={path/to/papi|OFF}` : Specify path to PAPI installed directory. The default is `OFF`
 - `-D REAL_TYPE={float|double}`: Specify real type of floating point. The default is `float`.
 - `-D with_ACC={no|yes}`
 - `-D TPR_PERF={no|yes}`: Enable Performance monitoring for each stage in TPR. This option may affect the performance.
 - `-D ENABLE_TEST={no|yes}`: Enable building test. The default is `no`
 - `-D BUILD_CUDA={no|yes}`: Build tpr.cu. The default is `no`
 - `-D CU_ARCH={CMAKE_CUDA_ARCHITECTURES}`: (cmake >= 3.18) CUDA architectures. ex) ITO-B(Tesla P100) -> "60"

## References
 - Adrián P. Diéguez, Margarita Amor, and Ramón Doallo. 2019. Tree Partitioning Reduction: A New Parallel Partition Method for Solving Tridiagonal Systems. ACM Trans. Math. Softw. 45, 3, Article 31 (August 2019), 26 pages. DOI:https://doi.org/10.1145/3328731

