#pragma once
#include <cassert>
#include <cmath>
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

// pure function
#ifdef __GNUC__
#define pure_function __attribute__((const))
#else
#define pure_function
#endif

#if (defined __amd64__) || (defined __amd64) || (defined __x86_64__) || \
    (defined __x86_64)
#define ILOG2_USE_x86_ASM
#endif

class Solver {
   public:
    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs);
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

static void print_array(real *array, int n);
#pragma acc routine seq
static int fllog2(int a) pure_function;
static inline uint32_t ilog2(const uint32_t x) pure_function;
#pragma acc routine seq
static int pow2(int k) pure_function;

static void print_array(real *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

#pragma acc routine seq
static int fllog2(int a) {
#ifdef ILOG2_USE_x86_ASM
    return (int)ilog2(static_cast<uint32_t>(a));
#else
    return (int)log2((double)a);
#endif
}

#ifdef ILOG2_USE_x86_ASM
/**
 * @brief floor(log_2(x)) for unsigned integer
 *
 * @note available for x86 or x86-64
 *
 * @param x
 * @return floor(log_2(x))
 */
static inline uint32_t ilog2(const uint32_t x) {
    uint32_t y;
    asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
    return y;
}
#endif

#pragma acc routine seq
static int pow2(int k) {
    assert(k >= 0);
    return 1 << k;
}

#undef pure_function
#undef ILOG2_USE_x86_ASM
