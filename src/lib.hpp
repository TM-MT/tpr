#pragma once
#include <iostream>


// precision
// same def as cz
#ifdef _REAL_IS_DOUBLE_
using real = double;
#else
/** 実数型の指定
 * - デフォルトでは、REAL_TYPE=float
 * - コンパイル時オプション-D_REAL_IS_DOUBLE_を付与することで
 *   REAL_TYPE=doubleになる
*/
using real = float;
#endif


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
