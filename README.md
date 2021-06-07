# Tree Partitioning Reduction

## Requirements
 - cmake >= 3.0.0
 - g++ >= 9.0.0
 - PAPI == papi-5-7-0-t, if using HWPC 

## Clone & Build & Run

```sh
$ git clone --recurse-submodules https://github.com/TM-MT/tpr
$ cd tpr
$ mkdir build && cd build
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make
# run sample code
$ ./src/tpr_main
# run with PMlib Reporting
$ ./src/tpr_pm
```

#### Options
 - `-D CMAKE_BUILD_TYPE={Release|Debug}`: Build type
 - `-D with_PAPI={path/to/papi|OFF}` : Specify path to PAPI installed directory. The default if `OFF`
 - `-D REAL_TYPE={float|double}`: Specify real type of floating point. The default is `float`.

## References
 - Adrián P. Diéguez, Margarita Amor, and Ramón Doallo. 2019. Tree Partitioning Reduction: A New Parallel Partition Method for Solving Tridiagonal Systems. ACM Trans. Math. Softw. 45, 3, Article 31 (August 2019), 26 pages. DOI:https://doi.org/10.1145/3328731

