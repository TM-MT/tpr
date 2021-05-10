#pragma once
#include <iostream>

using real = double;

class Solver
{
public:
    int solve();
    int get_ans(real *x);
};



template <typename T>
T max(T a, T b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

template <typename T>
T min(T a, T b) {
    if (a > b) {
        return b;
    } else {
        return a;
    }
}

void print_array(real *array, int n);
int fllog2(int a);
int pow2(int k);
