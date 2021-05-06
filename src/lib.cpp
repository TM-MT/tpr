#include <iostream>
#include "lib.hpp"



void print_array(real *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}
