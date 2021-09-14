#pragma once
// #include "lib.hpp"

__global__ void tpr_ker(float *a, float *b, float *c, float *x, int n, int s);
void tpr_cu(float *a, float *c, float *rhs, int n, int s) ;
