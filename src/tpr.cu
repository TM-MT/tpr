#include "tpr.cuh"
#include "main.hpp"
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

using namespace TPR_CU;



__global__ void tpr_ker(float *a, float *c, float *rhs, float *x, int n, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = idx / s * s;
    int ed = st + s - 1;

    Equation eq;
    eq.a = a;
    eq.c = c;
    eq.rhs = rhs;
    eq.x = x;

    TPR_Params params;
    params.n = n;
    params.s = s;
    params.idx = idx;
    params.st = st;
    params.ed = ed;

    cg::thread_block tb = cg::this_thread_block();

    float tmp_aa, tmp_cc, tmp_rr;
    float inter_ast, inter_cst, inter_rhsst; // bkup
    float inter_aed, inter_ced, inter_rhsed; // bkup


    tpr_st1_ker(tb, eq, params);


    // Update E_{st} by E_{ed}
    if (idx < n && idx == st) {
        int k = st, kr = ed;
        float ak = a[k], akr = a[kr];
        float ck = c[k], ckr = c[kr];
        float rhsk = rhs[k], rhskr = rhs[kr];

        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        tmp_aa = inv_diag_k * ak;
        tmp_cc = -inv_diag_k * ckr * ck;
        tmp_rr = inv_diag_k * (rhsk - rhskr * ck);

        inter_ast = a[st];
        inter_cst = c[st];
        inter_rhsst = rhs[st];

        // __syncthreads();

        a[idx] = tmp_aa;
        c[idx] = tmp_cc;
        rhs[idx] = tmp_rr;
    }



    // Update E_{st-1} by E_{st}
    if (idx < n - 1 && idx == ed) {
        int k = idx, kr = idx+1; // (k, kr) = (st-1, st)
        float ak = a[k], akr = a[kr];
        float ck = c[k], ckr = c[kr];
        float rhsk = rhs[k], rhskr = rhs[kr];
        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        inter_aed = a[idx];
        inter_ced = c[idx];
        inter_rhsed = rhs[idx];

        a[k] = inv_diag_k * ak;
        c[k] = -inv_diag_k * ckr * ck;
        rhs[k] = inv_diag_k * (rhsk - rhskr * ck);
    }

    // FIX-ME should be block sync

    // PCR
    for (int p = static_cast<int>(log2f(static_cast<double>(s))) + 1;
         p <= static_cast<int>(log2f(static_cast<double>(n)));
         p++)
    {
        if (idx < n && idx == ed) {
            // reduction
            int u = 1 << (p - 1); // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0;
                ckl = 0.0;
                rkl = 0.0;
            } else {
                akl = a[lidx];
                ckl = c[lidx];
                rkl = rhs[lidx];
            }
            int ridx = idx + u;
            float akr, ckr, rkr;
            if (ridx >= n) {
                akr = 0.0;
                ckr = -1.0;
                rkr = 0.0;
            } else {
                akr = a[ridx];
                ckr = c[ridx];
                rkr = rhs[ridx];
            }

            float inv_diag_k = 1.0 / (1.0 - ckl * a[idx] - akr * c[idx]);

            tmp_aa = - inv_diag_k * akl* a[idx];
            tmp_cc = - inv_diag_k * ckr * c[idx];
            tmp_rr = inv_diag_k * (rhs[idx] - rkl * a[idx] - rkr * c[idx]);
        }

        __syncthreads();

        if (idx < n && idx == ed) {
        // copy back
            a[idx] = tmp_aa;
            c[idx] = tmp_cc;
            rhs[idx] = tmp_rr;
        }

        __syncthreads();
    }

    __syncthreads();


    tpr_st2_copyback(tb, rhs, x, n, s);
    __syncthreads();
    // stage 3
    if (idx < n) {
        if (idx == st) {
            a[idx] = inter_ast;
            c[idx] = inter_cst;
            rhs[idx] = inter_rhsst;
        }

        if (idx == ed && idx != n-1) {
            a[idx] = inter_aed;
            c[idx] = inter_ced;
            rhs[idx] = inter_rhsed;
        }
    }
    __syncthreads();
    tpr_st3_ker(tb, eq, params);
 
    return ;
}


// stage 1
__device__ void tpr_st1_ker(cg::thread_block tb, Equation eq, TPR_Params params){
    int idx = params.idx;
    int n = params.n, s = params.s;
    int st = params.st, ed = params.ed;
    float tmp_aa, tmp_cc, tmp_rr;

    for (int p = 1; p <= static_cast<int>(log2f(static_cast<double>(s))); p++) {
        if (idx < n) {
            // reduction
            int u = 1 << (p - 1); // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < st) {
                akl = -1.0;
                ckl = 0.0;
                rkl = 0.0;
            } else {
                akl = eq.a[lidx];
                ckl = eq.c[lidx];
                rkl = eq.rhs[lidx];
            }
            int ridx = idx + u;
            float akr, ckr, rkr;
            if (ridx > ed) {
                akr = 0.0;
                ckr = -1.0;
                rkr = 0.0;
            } else {
                akr = eq.a[ridx];
                ckr = eq.c[ridx];
                rkr = eq.rhs[ridx];
            }

            float inv_diag_k = 1.0 / (1.0 - ckl * eq.a[idx] - akr * eq.c[idx]);

            tmp_aa = - inv_diag_k * akl* eq.a[idx];
            tmp_cc = - inv_diag_k * ckr * eq.c[idx];
            tmp_rr = inv_diag_k * (eq.rhs[idx] - rkl * eq.a[idx] - rkr * eq.c[idx]);
        }

        tb.sync();

        if (idx < n) {
            // copy back
            eq.a[idx] = tmp_aa;
            eq.c[idx] = tmp_cc;
            eq.rhs[idx] = tmp_rr;
        }

        tb.sync();
    }
}



// copy the answer from stage 2 PCR
__device__ void tpr_st2_copyback(cg::thread_block tb, float *rhs, float *x, int n, int s) {
	int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int st = idx / s * s;
    int ed = st + s - 1;

        if (idx < n && idx == ed) {
            x[idx] = rhs[idx];
        }
}

__device__ void tpr_st3_ker(cg::thread_block tb, Equation eq, TPR_Params params) {
    int idx = params.idx;
    int st = params.st;
    int ed = params.ed;
    int n = params.n;

   if (idx < n) {
        int lidx = max(0, st - 1);

        float key = 1.0 / eq.c[ed] * (eq.rhs[ed] - eq.a[ed] * eq.x[lidx] - eq.x[ed]);
        if (eq.c[ed] == 0.0) {
            key = 0.0;
        }

        eq.x[idx] = eq.rhs[idx] - eq.a[idx] * eq.x[lidx] - eq.c[idx] * key;
    }
    return ;
}


__global__ void pcr_ker(float *a, float *c, float *rhs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_aa, tmp_cc, tmp_rr;

    if (idx < n) {
        for (int p = 1; p <= static_cast<int>(log2f(static_cast<double>(n))); p++) {
            // reduction
            int u = 1 << (p - 1); // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0;
                ckl = 0.0;
                rkl = 0.0;
            } else {
                akl = a[lidx];
                ckl = c[lidx];
                rkl = rhs[lidx];
            }
            int ridx = idx + u;
            float akr, ckr, rkr;
            if (ridx >= n) {
                akr = 0.0;
                ckr = -1.0;
                rkr = 0.0;
            } else {
                akr = a[ridx];
                ckr = c[ridx];
                rkr = rhs[ridx];
            }

            float inv_diag_k = 1.0 / (1.0 - ckl * a[idx] - akr * c[idx]);

            tmp_aa = - inv_diag_k * akl* a[idx];
            tmp_cc = - inv_diag_k * ckr * c[idx];
            tmp_rr = inv_diag_k * (rhs[idx] - rkl * a[idx] - rkr * c[idx]);

            __syncthreads();

            // copy back
            a[idx] = tmp_aa;
            c[idx] = tmp_cc;
            rhs[idx] = tmp_rr;

            __syncthreads();
        }
    }
}



#define CU_CHECK( expr ) { cudaError_t t = expr;\
    if (t != cudaSuccess) {\
        fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(t), t, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
}


int main() {
    int n = 1024;
    struct TRIDIAG_SYSTEM *sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    for (int s = 128; s <= n; s *= 2) {
        assign(sys);
        tpr_cu(sys->a, sys->c, sys->rhs, n, s);
    }

    assign(sys);
    pcr_cu(sys->a, sys->c, sys->rhs, n);

    clean(sys);
    free(sys);

}


int setup(struct TRIDIAG_SYSTEM *sys, int n) {
    sys->a = (real *)malloc(n * sizeof(real));
    sys->diag = (real *)malloc(n * sizeof(real));
    sys->c = (real *)malloc(n * sizeof(real));
    sys->rhs = (real *)malloc(n * sizeof(real));
    sys->n = n;

    return sys_null_check(sys);
}

int assign(struct TRIDIAG_SYSTEM *sys) {
    int n = sys->n;
    for (int i = 0; i < n; i++) {
        sys->a[i] = -1.0/6.0;
        sys->c[i] = -1.0/6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = 1.0 * (i+1);
    }
    sys->a[0] = 0.0;
    sys->c[n-1] = 0.0;

    return 0;
}



int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        free(p);
    }

    sys->a = nullptr;
    sys->diag = nullptr;
    sys->c = nullptr;
    sys->rhs = nullptr;

    return 0;
}


bool sys_null_check(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}



void tpr_cu(float *a, float *c, float *rhs, int n, int s) {
    int size = n * sizeof(float);
    // Host
    float *x;

    x = (float*)malloc(size);

    // Device
    float *d_a, *d_c, *d_r;   // device copies of a, c, rhs
    float *d_x;
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));
    CU_CHECK(cudaMalloc((void **)&d_x, size));

    std::cerr << "TPR: s=" << s << "\n";
    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)); 
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    // launch
    tpr_ker<<<n / s, s>>>(d_a, d_c, d_r, d_x, n, s);

    cudaDeviceSynchronize();

    CU_CHECK(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        std::cout << x[i] << ", ";
    }
    std::cout << "\n";


    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    CU_CHECK(cudaFree(d_x));
    free(x);
    return ;
}



void pcr_cu(float *a, float *c, float *rhs, int n) {
    int size = n * sizeof(float);
    // Host
    float *x;

    x = (float*)malloc(size);

    // Device
    float *d_a, *d_c, *d_r;   // device copies of a, c, rhs
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));

    std::cerr << "PCR\n";
    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    pcr_ker<<<1, n>>>(d_a, d_c, d_r, n);

    cudaDeviceSynchronize();
    CU_CHECK(cudaMemcpy(x, d_r, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        std::cout << x[i] << ", ";
    }
    std::cout << "\n";

    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    free(x);
    return ;
}
