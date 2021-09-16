#include "tpr.cuh"
#include <iostream>


__global__ void tpr_ker(float *a, float *c, float *rhs, float *x, int n, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = n / s;
    int st = idx / s * s;
    int ed = st + s - 1;
    // printf("%d: %d, %d\n", idx, st, ed);

    float tmp_aa, tmp_cc, tmp_rr;
    float inter_ast, inter_cst, inter_rhsst; // bkup
    float inter_aed, inter_ced, inter_rhsed; // bkup

    // stage 1
    if (idx < n) {
        for (int p = 1; p <= static_cast<int>(log2f(static_cast<double>(s))); p++) {
            // reduction
            int u = 1 << (p - 1); // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < st) {
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
            if (ridx > ed) {
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

        // Update E_{st} by E_{ed}
        if (idx == st) {
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

            __syncthreads();

            a[st] = tmp_aa;
            c[st] = tmp_cc;
            rhs[st] = tmp_rr;
        }
        __syncthreads();
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
    __syncthreads();


    // PCR
    if (idx < m) {
        for (int p = static_cast<int>(log2f(static_cast<double>(s))) + 1;
             p <= static_cast<int>(log2f(static_cast<double>(n))); 
             p++) 
        {
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
            if (ridx >= m) {
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

    // FIX-ME should be block sync

    // stage 3
    if (idx < n) {
        // copy the answer from stage 2 PCR
        if (idx == ed) {
            x[idx] = rhs[idx];
        }

        // FIX-ME should be block sync
        __syncthreads();

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

        __syncthreads();

        int lidx = max(0, st - 1);

        float key = 1.0 / c[ed] * (rhs[ed] - a[ed] * x[lidx] - x[ed]);
        if (c[ed] == 0.0) {
            key = 0.0;
        }

        x[idx] = rhs[idx] - a[idx] * x[lidx] - c[idx] * key;
    }
}

/**
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
**/


#define CU_CHECK( expr ) { cudaError_t t = expr;\
    if (t != cudaSuccess) {\
        fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(t), t, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
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

    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)); 
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice)); 

    cudaDeviceSynchronize();

    // launch
    tpr_ker<<<n / s, s>>>(d_a, d_c, d_r, d_x, n, s);

    cudaDeviceSynchronize();
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost); 

    for (int i = 0; i < n; i++) {
        std::cout << x[i] << ", ";
    }
    std::cout << "\n";
    cudaFree(d_a);
    cudaFree(d_c);
    cudaFree(d_r);
    cudaFree(d_x);
    free(x);
}
