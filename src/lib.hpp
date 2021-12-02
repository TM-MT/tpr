#pragma once
#include <cassert>
#include <cmath>
#include <iostream>

#ifdef _REAL_IS_DOUBLE_
using real = double;
const double one = 1.0;
#else
using real = float;
const float one = 1.0f;
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
static int fllog2(int a) pure_function;
static inline uint32_t ilog2(const uint32_t x) pure_function;
static int pow2(int k) pure_function;
static int file_print_array(std::string &path, real *x, int n);
static int fprint_array(FILE *fp, real *x, int n);

static void print_array(real *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

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

static int pow2(int k) {
    assert(k >= 0);
    return 1 << k;
}

static int file_print_array(std::string &path, real *x, int n) {
    FILE *fp;
    fp = fopen(path.c_str(), "w");
    if (fp == nullptr) {
        fprintf(stderr, "[%s][Error] Failed to open `%s`\n", __FILE__,
                path.c_str());
        exit(EXIT_FAILURE);
    }
    fprint_array(fp, x, n);
    fclose(fp);
    return 0;
}

#ifdef _REAL_IS_DOUBLE_
#define CONVERSION_FORMAT "%lf"
#else
#define CONVERSION_FORMAT "%f"
#endif

#define DELIMITER ","
static int fprint_array(FILE *fp, real *x, int n) {
    for (int i = 0; i < n; i++) {
        fprintf(fp, CONVERSION_FORMAT, x[i]);
        if (i < n - 1) {
            fprintf(fp, DELIMITER);
        }
    }
    fprintf(fp, "\n");
    return 0;
}
#undef CONVERSION_FORMAT
#undef DELIMITER

#undef pure_function
#undef ILOG2_USE_x86_ASM
