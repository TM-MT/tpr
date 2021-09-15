#include "tpr.cuh"
#include <iostream>
#define SM_SIZE 256


__global__ void tpr_ker(float *a, float *c, float *rhs, float *x, int n, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = n / s;
    int st = idx / s * s;
    int ed = st + s - 1;

    float tmp_aa, tmp_cc, tmp_rr;
    float *inter_a, *inter_c, *inter_rhs;
    float *st2_a, *st2_c, *st2_rhs;
        // if ((inter_a == NULL) || (inter_c == NULL) || (inter_rhs == NULL)
        //     (st2_a == NULL) || (st2_c == NULL) || (st2_rhs == NULL)) {
        // }

    if (idx < n) {
        inter_a = (float*)malloc(2 * m * sizeof(float));
        inter_c = (float*)malloc(2 * m * sizeof(float));
        inter_rhs = (float*)malloc(2 * m * sizeof(float));
        st2_a = (float *)malloc(m * sizeof(float));
        st2_c = (float *)malloc(m * sizeof(float));
        st2_rhs = (float *)malloc(m * sizeof(float));


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
    }

    if (idx == st) {
        int k = st, kr = ed;
        int eqi_dst = 2 * st / s;
        float ak = a[k];
        float akr = a[kr];
        float ck = c[k];
        float ckr = c[kr];
        float rhsk = rhs[k];
        float rhskr = rhs[kr];

        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        inter_a[eqi_dst] = inv_diag_k * ak;
        inter_c[eqi_dst] = -inv_diag_k * ckr * ck;
        inter_rhs[eqi_dst] = inv_diag_k * (rhsk - rhskr * ck);

        // Copy E_{ed}
        inter_a[eqi_dst + 1] = akr; // a.k.a. a[ed]
        inter_c[eqi_dst + 1] = ckr;
        inter_rhs[eqi_dst + 1] = rhskr;
    }
    __syncthreads();

    if (1 <= idx && idx < 2 * m - 1 && idx % 2 == 1) {
        float ak = inter_a[idx];
        float akr = inter_a[idx + 1];
        float ck = inter_c[idx];
        float ckr = inter_c[idx + 1];
        float rhsk = inter_rhs[idx];
        float rhskr = inter_rhs[idx + 1];
        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        int dst = idx / 2;
        st2_a[dst] = inv_diag_k * ak;
        st2_c[dst] = -inv_diag_k * ckr * ck;
        st2_rhs[dst] = inv_diag_k * (rhsk - rhskr * ck);        
    }
    __syncthreads();

    // PCR
    if (idx < m) {
        for (int p = 1; p <= static_cast<int>(log2f(static_cast<double>(m))); p++) {
            // reduction
            int u = 1 << (p - 1); // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0;
                ckl = 0.0;
                rkl = 0.0;
            } else {
                akl = st2_a[lidx];
                ckl = st2_c[lidx];
                rkl = st2_rhs[lidx];
            }
            int ridx = idx + u;
            float akr, ckr, rkr;
            if (ridx >= m) {
                akr = 0.0;
                ckr = -1.0;
                rkr = 0.0;
            } else {
                akr = st2_a[ridx];
                ckr = st2_c[ridx];
                rkr = st2_rhs[ridx];
            }

            float inv_diag_k = 1.0 / (1.0 - ckl * st2_a[idx] - akr * st2_c[idx]);

            tmp_aa = - inv_diag_k * akl* st2_a[idx];
            tmp_cc = - inv_diag_k * ckr * st2_c[idx];
            tmp_rr = inv_diag_k * (st2_rhs[idx] - rkl * st2_a[idx] - rkr * st2_c[idx]);

            __syncthreads();

            // copy back
            st2_a[idx] = tmp_aa;
            st2_c[idx] = tmp_cc;
            st2_rhs[idx] = tmp_rr;

            __syncthreads();
        }
    }

    if (idx < n) {
        // for (int i = s - 1; i < n; i += s) {
        // copy the answer
        if (idx % s == s - 1) {
           x[idx] = st2_rhs[idx / s];
        }
        __syncthreads();

        int lidx = max(0, st - 1);

        float key = 1.0 / c[ed] * (rhs[ed] - a[ed] * x[lidx] - x[ed]);
        if (c[ed] == 0.0) {
            key = 0.0;
        }

        x[idx] = rhs[idx] - a[idx] * x[lidx] - c[idx] * key;
    }
    __syncthreads();
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
