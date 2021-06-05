# Tree Partitioning Reduction

## Build & Run

```sh
$ mkdir build && cd build
$ cmake ..
$ make
$ ./src/tpr_main
```

#### Options
 - `-D CMAKE_BUILD_TYPE={Release|Debug}`: Build type
 - `-D with_PAPI={path/to/papi|OFF}` : Specify path to PAPI installed directory
 - `-D REAL_TYPE={float|double}`: Specify real type of floating point. The default is float.

