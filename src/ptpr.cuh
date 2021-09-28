#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

// #include "lib.hpp"

namespace cg = cooperative_groups;

namespace PTPR_CU {
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

__global__ void tpr_ker(float *a, float *b, float *c, float *x, int n, int s);
__device__ void tpr_st1_ker(cg::thread_block &tb, PTPR_CU::Equation eq,
                            PTPR_CU::TPR_Params const &params);
__device__ void tpr_inter(cg::thread_block &tb, PTPR_CU::Equation eq,
                          float3 &bkup, PTPR_CU::TPR_Params const &params);
__device__ void tpr_inter_global(cg::thread_block &tb, PTPR_CU::Equation eq,
                                 float3 &bkup,
                                 PTPR_CU::TPR_Params const &params);
__device__ void tpr_st2_copyback(cg::thread_block &tb, float *rhs, float *x,
                                 int n, int s);
__device__ void tpr_st3_ker(cg::thread_block &tb, PTPR_CU::Equation eq,
                            PTPR_CU::TPR_Params const &params);
__global__ void pcr_ker(float *a, float *c, float *rhs, int n);
void ptpr_cu(float *a, float *c, float *rhs, int n, int s);
void pcr_cu(float *a, float *c, float *rhs, int n);
}  // namespace PTPR_CU
