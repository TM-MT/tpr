#include <iostream>
#include <cmath>
#include <assert.h>
#include "lib.hpp"



void print_array(real *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}


int fllog2(int a) {
    return (int)log2((double)a);
}


int pow2(int k) {
    assert(k >= 0);
    return 1 << k;
}
