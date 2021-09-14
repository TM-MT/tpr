# Tree Partitioning Reduction

## Requirements
 - cmake >= 3.8.0
 - g++ >= 9.0.0
 - PAPI == papi-5-7-0-t, if using HWPC

#### Requirements for GPU Run
 - nvc++ >= 21.5-0

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
$ cmake -D CMAKE_BUILD_TYPE=Release -Dwith_ACC=yes -DCMAKE_C_FLAGS="-noswitcherror -gpu=cc75" -DCMAKE_CXX_FLAGS="-noswitcherror -gpu=cc75" -DRandom_BuildTests=off  ..

# Use cuda
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_FLAGS="-Minfo=accel -noswitcherror -cuda -gpu=cc60" -DCMAKE_C_FLAGS="-noswitcherror" -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50"  -Dwith_ACC=no -DRandom_BuildTests=no ..

```

#### Options
 - `-D CMAKE_BUILD_TYPE={Release|Debug}`: Build type
 - `-D with_PAPI={path/to/papi|OFF}` : Specify path to PAPI installed directory. The default if `OFF`
 - `-D REAL_TYPE={float|double}`: Specify real type of floating point. The default is `float`.
 - `-D with_ACC={no|yes}`

## References
 - Adrián P. Diéguez, Margarita Amor, and Ramón Doallo. 2019. Tree Partitioning Reduction: A New Parallel Partition Method for Solving Tridiagonal Systems. ACM Trans. Math. Softw. 45, 3, Article 31 (August 2019), 26 pages. DOI:https://doi.org/10.1145/3328731

