#include <bits/stdint-uintn.h>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <stdint.h>

#include "lib.hpp"

#if (defined __amd64__) || (defined __amd64) || (defined __x86_64__) || (defined __x86_64)
#define ILOG2_USE_x86_ASM
#pragma message("USING x86 ASM")
#endif



void print_array(real *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}


int fllog2(int a) {
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
inline uint32_t ilog2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}
#endif

int pow2(int k) {
    assert(k >= 0);
    return 1 << k;
}

#undef ILOG2_USE_x86_ASM
