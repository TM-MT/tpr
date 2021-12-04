#pragma once
#ifdef __NVCC__
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <array>
#include <tuple>

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
__device__ void tpr_st1_ker(cg::thread_block &tb, TPR_CU::Equation &eq,
                            TPR_CU::TPR_Params const &params);
__device__ void tpr_inter(cg::thread_block &tb, TPR_CU::Equation &eq,
                          TPR_CU::TPR_Params const &params);
__device__ void tpr_inter_global(cg::thread_block &tb, TPR_CU::Equation &eq,
                                 TPR_CU::TPR_Params const &params,
                                 float *pbuffer);
__device__ void tpr_st2_ker(cg::grid_group &tg, cg::thread_block &tb,
                            TPR_CU::Equation &eq, TPR_CU::TPR_Params &params,
                            float *pbuffer);
__device__ void tpr_st3_ker(cg::thread_block &tb, TPR_CU::Equation &eq,
                            TPR_CU::TPR_Params const &params);
__global__ void cr_ker(float *a, float *c, float *rhs, float *x, int n);
__device__ void cr_thread_block(cg::thread_block &tb, float *a, float *c,
                                float *rhs, float *x, int n);
std::tuple<dim3, dim3, size_t> tpr_launch_config(int n, int s, int dev);
std::array<dim3, 2> n2dim(int n, int s, int dev);

__device__ long int get_time();
}  // namespace TPR_CU
#endif
namespace TPR_CU {
void tpr_cu(float *a, float *c, float *rhs, float *x, int n, int s);
void cr_cu(float *a, float *c, float *rhs, float *x, int n);
#ifdef _REAL_IS_DOUBLE_
void tpr_cu(double *a, double *c, double *rhs, double *x, int n, int s);
void cr_cu(double *a, double *c, double *rhs, double *x, int n);
#endif
}  // namespace TPR_CU
