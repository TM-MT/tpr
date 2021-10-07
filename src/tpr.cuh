#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <array>
#include <tuple>

#include "pm.cuh"

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
    int m;
    int idx;
    int st;
    int ed;
};

__global__ void tpr_ker(float *a, float *b, float *c, float *x, float *pbuffer,
                        int n, int s);
__device__ void tpr_st1_ker(cg::thread_block &tb, TPR_CU::Equation eq,
                            TPR_CU::TPR_Params const &params);
__device__ void tpr_inter(cg::thread_block &tb, TPR_CU::Equation eq,
                          TPR_CU::TPR_Params const &params);
__device__ void tpr_inter_global(cg::thread_block &tb, TPR_CU::Equation eq,
                                 TPR_CU::TPR_Params const &params);
__device__ void tpr_st3_ker(cg::thread_block &tb, TPR_CU::Equation eq,
                            TPR_CU::TPR_Params const &params);
__global__ void cr_ker(float *a, float *c, float *rhs, float *x, int n);
void tpr_cu(float *a, float *c, float *rhs, float *x, int n, int s);
std::tuple<dim3, dim3, size_t> tpr_launch_config(int n, int s, int dev);
std::array<dim3, 2> n2dim(int n, int s, int dev);
void cr_cu(float *a, float *c, float *rhs, float *x, int n);
}  // namespace TPR_CU
