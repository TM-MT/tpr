#include "tpr.cuh"
#include <iostream>
#define SM_SIZE 256


__global__ void tpr_ker(float *a, float *c, float *rhs, float *x, int n, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = n / s;
    int st = idx / s * s;
    int ed = st + s - 1;

    float tmp_aa, tmp_cc, tmp_rr;

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
    }
}


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
