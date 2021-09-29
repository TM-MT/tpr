#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <iostream>

#include "main.hpp"
#include "tpr.cuh"

namespace cg = cooperative_groups;

using namespace TPR_CU;

/**
 * for dynamic shared memory use
 */
extern __shared__ float array[];

/**
 * @brief      TPR main kernel
 *
 * @param      a     { parameter_description }
 * @param      c     { parameter_description }
 * @param      rhs   The right hand side
 * @param      x     { parameter_description }
 * @param[in]  n     { parameter_description }
 * @param[in]  s     { parameter_description }
 */
__global__ void TPR_CU::tpr_ker(float *a, float *c, float *rhs, float *x, int n,
                                int s) {
    cg::grid_group tg = cg::this_grid();
    cg::thread_block tb = cg::this_thread_block();
    assert(tg.is_valid());
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = idx / s * s;
    int ed = st + s - 1;

    // local copy
    // sha[0:s], shc[0:s], shrhs[0:s]
    __shared__ float *sha, *shc, *shrhs;
    sha = (float *)array;
    shc = (float *)&array[s];
    shrhs = (float *)&array[2 * s];

    // make local copy on shared memory
    cg::memcpy_async(tb, sha, &a[st], sizeof(float) * s);
    cg::memcpy_async(tb, shc, &c[st], sizeof(float) * s);
    cg::memcpy_async(tb, shrhs, &rhs[st], sizeof(float) * s);

    Equation eq;
    eq.a = sha;
    eq.c = shc;
    eq.rhs = shrhs;
    eq.x = x;

    TPR_Params params;
    params.n = n;
    params.s = s;
    params.idx = idx;
    params.st = st;
    params.ed = ed;

    float tmp_aa, tmp_cc, tmp_rr;
    // bkups, .x -> a, .y -> c, .z -> rhs
    float3 bkup;

    cg::wait(tb);

    // TPR Stage 1
    if (idx < n && idx % 2 == 0) {
        bkup.x = sha[idx - st];
        bkup.y = shc[idx - st];
        bkup.z = shrhs[idx - st];
    }

    tpr_st1_ker(tb, eq, params);

    if (idx < n && idx % 2 == 1) {
        bkup.x = sha[idx - st];
        bkup.y = shc[idx - st];
        bkup.z = shrhs[idx - st];
    }
    tb.sync();

    tpr_inter(tb, eq, params);

    // copy back
    // since `tpr_inter_global` and stage 2 are global operations,
    // eq.* should hold address in global memory
    a[idx] = sha[idx - st];
    c[idx] = shc[idx - st];
    rhs[idx] = shrhs[idx - st];
    eq.a = a;
    eq.c = c;
    eq.rhs = rhs;

    tg.sync();

    tpr_inter_global(tb, eq, params);

    tg.sync();

    // CR (TPR Stage 2)
    // CR Forward Reduction
    for (int p = static_cast<int>(log2f(static_cast<double>(s))) + 1;
         p <= static_cast<int>(log2f(static_cast<double>(n))); p++) {
        int u = 1 << (p - 1);  // offset
        int ux = 1 << (p + 1);
        bool select_idx =
            (idx < n) && ((idx - ux + 1) % ux == 0) && ((idx - ux + 1) >= 0);

        if (select_idx) {
            // reduction
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

            tmp_aa = -inv_diag_k * akl * a[idx];
            tmp_cc = -inv_diag_k * ckr * c[idx];
            tmp_rr = inv_diag_k * (rhs[idx] - rkl * a[idx] - rkr * c[idx]);
        }

        tg.sync();

        if (select_idx) {
            // copy back
            a[idx] = tmp_aa;
            c[idx] = tmp_cc;
            rhs[idx] = tmp_rr;
        }

        tg.sync();
    }

    // CR Intermediate
    if ((n > 1) && (idx == n / 2 - 1)) {
        int u = n / 2;
        float inv_det = 1.0 / (1.0 - c[idx] * a[idx + u]);

        x[idx] = (rhs[idx] - c[idx] * rhs[idx + u]) * inv_det;
        x[idx + u] = (rhs[idx + u] - rhs[idx] * a[idx + u]) * inv_det;
    }

    tg.sync();

    // CR Backward Substitution
    for (int p = static_cast<int>(log2f(static_cast<double>(n))) - 2;
         p >= static_cast<int>(log2f(static_cast<double>(s))); p--) {
        int u = 1 << p;
        int ux = 1 << (p + 1);

        if ((idx < n) && ((idx - u + 1) % ux == 0) && (idx - u + 1 >= 0)) {
            int lidx = idx - u;
            float x_u;
            if (lidx < 0) {
                x_u = 0.0;
            } else {
                x_u = x[lidx];
            }
            x[idx] = rhs[idx] - a[idx] * x_u - c[idx] * x[idx + u];
        }

        tg.sync();
    }
    // CR END

    tb.sync();
    // TPR stage 3
    if (idx < n) {
        sha[idx - st] = bkup.x;
        shc[idx - st] = bkup.y;
        shrhs[idx - st] = bkup.z;
    }

    tg.sync();

    // tpr_st3_ker use shared memory
    eq.a = sha;
    eq.c = shc;
    eq.rhs = shrhs;
    tpr_st3_ker(tb, eq, params);

    return;
}

/**
 * @brief      TPR Stage 1
 *
 * @param          tb      cg::thread_block
 * @param[in,out]  eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 * shared memory
 * @param[in]      params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_st1_ker(cg::thread_block &tb, Equation eq,
                                    TPR_Params const &params) {
    int idx = params.idx;
    int i = idx - params.st;
    int n = params.n, s = params.s;
    float tmp_aa, tmp_cc, tmp_rr;
    float *sha = eq.a, *shc = eq.c, *shrhs = eq.rhs;
    assert(__isShared((void *)sha));
    assert(__isShared((void *)shc));
    assert(__isShared((void *)shrhs));

    for (int p = 1; p <= static_cast<int>(log2f(static_cast<double>(s))); p++) {
        int u = 1 << (p - 1);  // offset
        int p2k = 1 << p;
        bool select_idx = idx < n && ((i % p2k == 0) || ((i + 1) % p2k == 0));

        if (select_idx) {
            // reduction
            int lidx = i - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0;
                ckl = 0.0;
                rkl = 0.0;
            } else {
                akl = sha[lidx];
                ckl = shc[lidx];
                rkl = shrhs[lidx];
            }
            int ridx = i + u;
            float akr, ckr, rkr;
            if (ridx >= s) {
                akr = 0.0;
                ckr = -1.0;
                rkr = 0.0;
            } else {
                akr = sha[ridx];
                ckr = shc[ridx];
                rkr = shrhs[ridx];
            }

            float inv_diag_k = 1.0 / (1.0 - ckl * sha[i] - akr * shc[i]);

            tmp_aa = -inv_diag_k * akl * sha[i];
            tmp_cc = -inv_diag_k * ckr * shc[i];
            tmp_rr = inv_diag_k * (shrhs[i] - rkl * sha[i] - rkr * shc[i]);
        }

        tb.sync();

        if (select_idx) {
            // copy back
            sha[i] = tmp_aa;
            shc[i] = tmp_cc;
            shrhs[i] = tmp_rr;
        }

        tb.sync();
    }
}

/**
 * @brief      TPR Intermediate stage 1
 *
 * Update E_{st} by E_{ed}
 *
 * @param          tb      cg::thread_block
 * @param[in,out]  eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 * shared memory
 * @param[out]     bkup    The bkup for stage 3 use. bkup->x: a, bkup->y: c,
 * bkup->z: rhs
 * @param[in]      params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_inter(cg::thread_block &tb, Equation eq,
                                  TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    float tmp_aa, tmp_cc, tmp_rr;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));

    if ((idx < params.n) && (idx == params.st)) {
        int k = idx - params.st, kr = params.s - 1;
        float ak = eq.a[k], akr = eq.a[kr];
        float ck = eq.c[k], ckr = eq.c[kr];
        float rhsk = eq.rhs[k], rhskr = eq.rhs[kr];

        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        tmp_aa = inv_diag_k * ak;
        tmp_cc = -inv_diag_k * ckr * ck;
        tmp_rr = inv_diag_k * (rhsk - rhskr * ck);

        eq.a[k] = tmp_aa;
        eq.c[k] = tmp_cc;
        eq.rhs[k] = tmp_rr;
    }
}

/**
 * @brief      TPR Intermediate stage GLOBAL
 *
 * Update E_{st-1} by E_{st}
 *
 * @param          tb      cg::thread_block
 * @param[in,out]  eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 * GLOBAL memory
 * @param[out]     bkup    The bkup for stage 3 use. bkup->x: a, bkup->y: c,
 * bkup->z: rhs
 * @param[in]      params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_inter_global(cg::thread_block &tb, Equation eq,
                                         TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int ed = params.ed;

    if ((idx < params.n - 1) && (idx == ed)) {
        int k = idx, kr = idx + 1;  // (k, kr) = (st-1, st)
        float ak = eq.a[k], akr = eq.a[kr];
        float ck = eq.c[k], ckr = eq.c[kr];
        float rhsk = eq.rhs[k], rhskr = eq.rhs[kr];
        float inv_diag_k = 1.0 / (1.0 - akr * ck);

        eq.a[k] = inv_diag_k * ak;
        eq.c[k] = -inv_diag_k * ckr * ck;
        eq.rhs[k] = inv_diag_k * (rhsk - rhskr * ck);
    }
}

/**
 * @brief      TPR Stage 3
 *
 * @param          tb      cg::thread_block
 * @param[in,out]  eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 * shared memory
 * @param[in]      params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_st3_ker(cg::thread_block &tb, Equation eq,
                                    TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int i = tb.thread_index().x;
    int st = params.st;
    int n = params.n, s = params.s;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));
    assert(__isGlobal((void *)eq.x));

    for (int p = static_cast<int>(log2f(static_cast<double>(s))) - 1; p >= 0;
         p--) {
        int u = 1 << p;

        if (idx < n && ((idx - st - u + 1) % (2 * u) == 0) &&
            ((idx - st - u + 1) >= 0)) {
            int lidx = idx - u;
            float x_u;
            if (lidx < 0) {
                x_u = 0.0;
            } else {
                x_u = eq.x[lidx];
            }

            eq.x[idx] = eq.rhs[i] - eq.a[i] * x_u - eq.c[i] * eq.x[idx + u];
        }
        tb.sync();
    }
    return;
}

__global__ void TPR_CU::cr_ker(float *a, float *c, float *rhs, float *x,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_aa, tmp_cc, tmp_rr;

    for (int p = 0; p < static_cast<int>(log2f(static_cast<double>(n))) - 1;
         p++) {
        int u = 1 << p;  // offset
        int ux = 1 << (p + 1);
        bool condition =
            (idx < n) && ((idx - ux + 1) % ux == 0) && ((idx - ux + 1) >= 0);

        // reduction
        if (condition) {
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

            tmp_aa = -inv_diag_k * akl * a[idx];
            tmp_cc = -inv_diag_k * ckr * c[idx];
            tmp_rr = inv_diag_k * (rhs[idx] - rkl * a[idx] - rkr * c[idx]);
        }

        __syncthreads();

        if (condition) {
            a[idx] = tmp_aa;
            c[idx] = tmp_cc;
            rhs[idx] = tmp_rr;
        }

        __syncthreads();
    }

    if ((n > 1) && (idx == n / 2 - 1)) {
        int u = n / 2;
        float inv_det = 1.0 / (1.0 - c[idx] * a[idx + u]);

        x[idx] = (rhs[idx] - c[idx] * rhs[idx + u]) * inv_det;
        x[idx + u] = (rhs[idx + u] - rhs[idx] * a[idx + u]) * inv_det;
    }

    __syncthreads();

    for (int p = static_cast<int>(log2f(static_cast<double>(n))) - 2; p >= 0;
         p--) {
        int u = 1 << p;
        int ux = 1 << (p + 1);

        if ((idx < n) && ((idx - u + 1) % ux == 0) && (idx - u + 1 >= 0)) {
            int lidx = idx - u;
            float x_u;
            if (lidx < 0) {
                x_u = 0.0;
            } else {
                x_u = x[lidx];
            }
            x[idx] = rhs[idx] - a[idx] * x_u - c[idx] * x[idx + u];
        }

        __syncthreads();
    }
    return;
}

#define CU_CHECK(expr)                                                     \
    {                                                                      \
        cudaError_t t = expr;                                              \
        if (t != cudaSuccess) {                                            \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", \
                    cudaGetErrorString(t), t, __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

int main() {
    int n = 8192;
    struct TRIDIAG_SYSTEM *sys =
        (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    TPR_ANS ans1(n), ans2(n);
    for (int s = 128; s <= std::min(n, 1024); s *= 2) {
        assign(sys);
        ans1.s = s;
        tpr_cu(sys->a, sys->c, sys->rhs, ans1.x, n, s);

        if (s > 128 && ans1 != ans2) {
            std::cout << "TPR(" << ans1.n << "," << ans1.s << ") and TPR("
                      << ans2.n << "," << ans2.s << ") has different answer.\n";
            std::cout << "TPR(" << ans1.n << "," << ans1.s << ")\n";
            ans1.display(std::cout);
            std::cout << "TPR(" << ans2.n << "," << ans2.s << ")\n";
            ans2.display(std::cout);
        }
        ans2 = ans1;
    }

    assign(sys);
    if (n <= 1024) {
        // currently CR works in thread,
        cr_cu(sys->a, sys->c, sys->rhs, ans1.x, n);
        ans1.display(std::cout);
    }

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
        sys->a[i] = -1.0 / 6.0;
        sys->c[i] = -1.0 / 6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = 1.0 * (i + 1);
    }
    sys->a[0] = 0.0;
    sys->c[n - 1] = 0.0;

    return 0;
}

int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        free(p);
    }

    sys->a = nullptr;
    sys->diag = nullptr;
    sys->c = nullptr;
    sys->rhs = nullptr;

    return 0;
}

bool sys_null_check(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}

/**
 * @brief      Helper function for tpr_cu
 *
 * 1. check if device support cooperative launch
 * 2. allocate device memory for compute
 * 3. launch kernel `tpr_cu`
 * 4. copy the answer from device to host
 * 6. free device memory
 *
 * @param[in]  a     { parameter_description }
 * @param[in]  c     { parameter_description }
 * @param[in]  rhs   The right hand side
 * @param[out] x     x[0:n] for the answer
 * @param[in]  n     { parameter_description }
 * @param[in]  s     { parameter_description }
 */
void TPR_CU::tpr_cu(float *a, float *c, float *rhs, float *x, int n, int s) {
    int dev = 0;
    int size = n * sizeof(float);

    // Device
    float *d_a, *d_c, *d_r;  // device copies of a, c, rhs
    float *d_x;
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));
    CU_CHECK(cudaMalloc((void **)&d_x, size));

    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    // launch configuration
    void *kernel_args[] = {&d_a, &d_c, &d_r, &d_x, &n, &s};
    auto config = tpr_launch_config(n, s, dev);
    // auto [dim_grid, dim_block, shmem_size] = rhs; not supported
    auto dim_grid = std::get<0>(config);
    auto dim_block = std::get<1>(config);
    auto shmem_size = std::get<2>(config);

    std::cerr << "TPR: s=" << s << "\n";
    // launch
    CU_CHECK(cudaLaunchCooperativeKernel((void *)tpr_ker, dim_grid, dim_block,
                                         kernel_args, shmem_size));

    cudaDeviceSynchronize();

    CU_CHECK(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));

    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    CU_CHECK(cudaFree(d_x));
    return;
}

/**
 * @brief launch configuration for tpr_ker
 * @details calculate suitable dimension and shared memory size for tpr_ker
 *
 * @param[in]  n     size of the equation
 * @param[in]  s     TPR parameter
 * @param[in]  dev   cuda device id
 * @return     [dim_grid, dim_block, shared_memory_size]
 */
std::tuple<dim3, dim3, size_t> TPR_CU::tpr_launch_config(int n, int s,
                                                         int dev) {
    // check cooperative launch support
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
                           dev);
    if (supportsCoopLaunch != 1) {
        printf("Cooperative launch not supported on dev %d.\n", dev);
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // calculate dimension
    auto dim = n2dim(n, s, dev);
    auto dim_grid = dim[0];
    auto dim_block = dim[1];

    size_t shmem_size = 4 * dim_block.x * sizeof(float);
    assert(shmem_size <= deviceProp.sharedMemPerBlock);

    std::tuple<dim3, dim3, size_t> ret(dim_grid, dim_block, shmem_size);
    return ret;
}

/**
 * @brief Helper function for tpr_cu
 * @details calculate dimension for cuda kernel launch.
 *
 * @param[in]  n     size of the equation
 * @param[in]  s     TPR parameter
 * @param[in]  dev   cuda device id
 * @return     [dim_grid, dim_block]
 */
std::array<dim3, 2> TPR_CU::n2dim(int n, int s, int dev) {
    assert(s > 0);
    assert(n >= s);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    auto max_tpb = deviceProp.maxThreadsPerBlock;

    if (s > max_tpb) {
        std::cerr << "TPR Parameter `s=" << s
                  << "` exceeds max threads per block: " << max_tpb << "\n";
        exit(EXIT_FAILURE);
    }

    dim3 dim_grid(n / s, 1, 1);  // we know `n >= s`
    dim3 dim_block(s, 1, 1);
    dim_grid.y = std::max(s / max_tpb, 1);

    return {dim_grid, dim_block};
}

void TPR_CU::cr_cu(float *a, float *c, float *rhs, float *x, int n) {
    int size = n * sizeof(float);

    // Device
    float *d_a, *d_c, *d_r, *d_x;  // device copies of a, c, rhs
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));
    CU_CHECK(cudaMalloc((void **)&d_x, size));

    std::cerr << "CR\n";
    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    cr_ker<<<1, n>>>(d_a, d_c, d_r, d_x, n);

    cudaDeviceSynchronize();
    CU_CHECK(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));

    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    CU_CHECK(cudaFree(d_x));
    return;
}
