#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

// #include "lib.hpp"

namespace cg = cooperative_groups;

namespace TPR_CU {
    struct Equation {
        float *a;
        float *c;
        float *rhs;
        float *x;
    };

    struct TPR_Params {
        int n;
        int s;
        int idx;
        int st;
        int ed;
    };
}

__global__ void tpr_ker(float *a, float *b, float *c, float *x, int n, int s);
__device__ void tpr_st1_ker(cg::thread_block tb, TPR_CU::Equation eq, TPR_CU::TPR_Params params);
__device__ void tpr_st2_copyback(cg::thread_block tb, float *rhs, float *x, int n, int s);
__device__ void tpr_st3_ker(cg::thread_block tb, TPR_CU::Equation eq, TPR_CU::TPR_Params params);
__global__ void pcr_ker(float *a, float *c, float *rhs, int n);
void tpr_cu(float *a, float *c, float *rhs, int n, int s) ;
void pcr_cu(float *a, float *c, float *rhs, int n);
