#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <iostream>

#include "main.hpp"
#include "tpr.cuh"

#ifdef TPR_PERF
#include "pm.cuh"
#endif

#if (__CUDACC_VER_MAJOR__ <= 11) && (__CUDACC_VER_MINOR__ < 4)
#pragma message("Using Experimental Features")
#define EXPERIMENTAL_ASYNC_COPY
#endif

namespace cg = cooperative_groups;
#ifdef EXPERIMENTAL_ASYNC_COPY
using namespace nvcuda::experimental;
#endif

using namespace TPR_CU;

/**
 * for dynamic shared memory use
 */
extern __shared__ float array[];

/**
 * @brief      TPR main kernel of TPR
 *
 * @param[in]  a        a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c        c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                      0.0
 * @param[in]  rhs      rhs[0:n] The right-hand-side of the equation.
 * @param[out] x        x[0:n] for the solution
 * @param      pbuffer  Additional memory for Stage 2 use. pbuffer[0:4 * n / s]
 * @param[in]  n        The order of A. `n` should be power of 2
 * @param[in]  s        The parameter of TPR. Each block handles `s` equations.
 */
__global__ void TPR_CU::tpr_ker(float *a, float *c, float *rhs, float *x,
                                float *pbuffer, int n, int s) {
    cg::grid_group tg = cg::this_grid();
    cg::thread_block tb = cg::this_thread_block();
    assert(tg.is_valid());
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = idx / s * s;

    // local copy
    // sha[0:s], shc[0:s], shrhs[0:s]
    __shared__ float *sha, *shc, *shrhs;
    sha = (float *)array;
    shc = (float *)&array[s];
    shrhs = (float *)&array[2 * s];

    // make local copy on shared memory
#ifdef EXPERIMENTAL_ASYNC_COPY
    pipeline pipe;
    memcpy_async(sha[idx - st], a[idx], pipe);
    memcpy_async(shc[idx - st], c[idx], pipe);
    memcpy_async(shrhs[idx - st], rhs[idx], pipe);
#else
    cg::memcpy_async(tb, sha, &a[st], sizeof(float) * s);
    cg::memcpy_async(tb, shc, &c[st], sizeof(float) * s);
    cg::memcpy_async(tb, shrhs, &rhs[st], sizeof(float) * s);
#endif

    Equation eq;
    eq.a = sha;
    eq.c = shc;
    eq.rhs = shrhs;
    eq.x = x;

    TPR_Params params;
    params.n = n;
    params.s = s;
    params.m = n / s;
    params.idx = idx;
    params.st = st;
    params.ed = st + s - 1;

    // bkups for stage 3 use, .x -> a, .y -> c, .z -> rhs
    float3 bkup;

#ifdef EXPERIMENTAL_ASYNC_COPY
    pipe.commit_and_wait();
#else
    cg::wait(tb);
#endif

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
    // since stage 2 are global operations, eq.* should hold address in global
    // memory
    a[idx] = sha[idx - st];
    c[idx] = shc[idx - st];
    rhs[idx] = shrhs[idx - st];
    eq.a = a;
    eq.c = c;
    eq.rhs = rhs;

    // make sure stage 1 operations have done.
    tg.sync();

    tpr_st2_ker(tg, tb, eq, params, pbuffer);

    tg.sync();

#ifdef EXPERIMENTAL_ASYNC_COPY
    if (blockIdx.x == 0 && idx < params.m) {
        pipe.commit_and_wait();
    }
#else
    if (blockIdx.x == 0) {
        cg::wait(tb);
    }
#endif

    // TPR stage 3
    if (idx < n) {
        sha[idx - st] = bkup.x;
        shc[idx - st] = bkup.y;
        shrhs[idx - st] = bkup.z;
    }

    tb.sync();

    // tpr_st3_ker use shared memory
    eq.a = sha;
    eq.c = shc;
    eq.rhs = shrhs;
    tpr_st3_ker(tb, eq, params);

    return;
}

/**
 * @brief         TPR Stage 1
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory
 * @param[in]     params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_st1_ker(cg::thread_block &tb, Equation &eq,
                                    TPR_Params const &params) {
    int idx = params.idx;
    int i = idx - params.st;
    int s = params.s;
    float tmp_aa, tmp_cc, tmp_rr;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));

    for (int p = 1; p <= static_cast<int>(log2f(static_cast<float>(s))); p++) {
        int p2k = 1 << p;
        bool select_idx =
            idx < params.n && ((i % p2k == 0) || ((i + 1) % p2k == 0));

        if (select_idx) {
            int u = 1 << (p - 1);  // offset

            // reduction
            float akl, ckl, rkl;
            {
                int lidx = i - u;
                if (lidx < 0) {
                    akl = -1.0f;
                    ckl = 0.0f;
                    rkl = 0.0f;
                } else {
                    akl = eq.a[lidx];
                    ckl = eq.c[lidx];
                    rkl = eq.rhs[lidx];
                }
            }
            float akr, ckr, rkr;
            {
                int ridx = i + u;
                if (ridx >= s) {
                    akr = 0.0f;
                    ckr = -1.0f;
                    rkr = 0.0f;
                } else {
                    akr = eq.a[ridx];
                    ckr = eq.c[ridx];
                    rkr = eq.rhs[ridx];
                }
            }

            float inv_diag_k = 1.0f / (1.0f - ckl * eq.a[i] - akr * eq.c[i]);

            tmp_aa = -inv_diag_k * akl * eq.a[i];
            tmp_cc = -inv_diag_k * ckr * eq.c[i];
            tmp_rr = inv_diag_k * (eq.rhs[i] - rkl * eq.a[i] - rkr * eq.c[i]);
        }

        tb.sync();

        if (select_idx) {
            // copy back
            eq.a[i] = tmp_aa;
            eq.c[i] = tmp_cc;
            eq.rhs[i] = tmp_rr;
        }

        tb.sync();
    }
}

/**
 * @brief         TPR Intermediate stage 1
 *
 *                Update E_{st} by E_{ed}
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory
 * @param[in]     params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_inter(cg::thread_block &tb, Equation &eq,
                                  TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));

    if ((idx < params.n) && (idx == params.st)) {
        int k = idx - params.st, kr = params.s - 1;
        float ak = eq.a[k], akr = eq.a[kr];
        float ck = eq.c[k], ckr = eq.c[kr];
        float rhsk = eq.rhs[k], rhskr = eq.rhs[kr];

        float inv_diag_k = 1.0f / (1.0f - akr * ck);

        eq.a[k] = inv_diag_k * ak;
        eq.c[k] = -inv_diag_k * ckr * ck;
        eq.rhs[k] = inv_diag_k * (rhsk - rhskr * ck);
    }
}

/**
 * @brief         TPR Intermediate stage GLOBAL
 *
 *                Update E_{st-1} by E_{st}
 *
 * @param         tb       cg::thread_block
 * @param[in,out] eq       Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                         GLOBAL memory
 * @param[in]     params   The parameters of TPR
 * @param         pbuffer  Additional memory for stage 2 use.
 */
__device__ void TPR_CU::tpr_inter_global(cg::thread_block &tb, Equation &eq,
                                         TPR_Params const &params,
                                         float *pbuffer) {
    assert(__isGlobal((void *)eq.a));
    assert(__isGlobal((void *)eq.c));
    assert(__isGlobal((void *)eq.rhs));
    int idx = params.idx;
    int ed = params.ed;

    if ((idx < params.n - 1) && (idx == ed)) {  // Update E_{st-1} by E_{st}
        int dst = idx / params.s;
        int k = idx, kr = idx + 1;  // (k, kr) = (st-1, st)
        float ak = eq.a[k], akr = eq.a[kr];
        float ck = eq.c[k], ckr = eq.c[kr];
        float rhsk = eq.rhs[k], rhskr = eq.rhs[kr];
        float inv_diag_k = 1.0f / (1.0f - akr * ck);

        pbuffer[dst] = inv_diag_k * ak;                    // a[k]
        pbuffer[params.m + dst] = -inv_diag_k * ckr * ck;  // c[k]
        pbuffer[2 * params.m + dst] =
            inv_diag_k * (rhsk - rhskr * ck);  // rhs[k]
    } else if (idx == params.n - 1) {          // copy E_{n-1} to pbuffer
        pbuffer[params.m - 1] = eq.a[idx];
        pbuffer[2 * params.m - 1] = eq.c[idx];
        pbuffer[3 * params.m - 1] = eq.rhs[idx];
    }
}

/**
 * @brief      TPR Stage 2
 *
 *             Call CR
 *
 * @param      tg       cg::grid_group
 * @param      tb       cg::thread_group
 * @param[in]  eq       Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                      GLOBAL memory
 * @param[in]  params   The parameters of TPR
 * @param      pbuffer  Additional global memory for stage 2
 */
__device__ void TPR_CU::tpr_st2_ker(cg::grid_group &tg, cg::thread_block &tb,
                                    Equation &eq, TPR_Params &params,
                                    float *pbuffer) {
    tpr_inter_global(tb, eq, params, pbuffer);

    tg.sync();

    if (blockIdx.x == 0) {
        int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
        int s = params.s, m = params.m;
        assert(m <= s);

        __shared__ float *sha, *shc, *shrhs, *shx;
        sha = (float *)array;
        shc = (float *)&array[s];
        shrhs = (float *)&array[2 * s];
        shx = (float *)&array[3 * s];

#ifdef EXPERIMENTAL_ASYNC_COPY
        pipeline pipe;
        if (idx < m) {
            memcpy_async(sha[idx - params.st], pbuffer[idx], pipe);
            memcpy_async(shc[idx - params.st], pbuffer[m + idx], pipe);
            memcpy_async(shrhs[idx - params.st], pbuffer[2 * m + idx], pipe);
            memcpy_async(shx[idx - params.st], pbuffer[3 * m + idx], pipe);
            pipe.commit_and_wait();
        }
#else
        cg::memcpy_async(tb, sha, &pbuffer[0], sizeof(float) * m);
        cg::memcpy_async(tb, shc, &pbuffer[m], sizeof(float) * m);
        cg::memcpy_async(tb, shrhs, &pbuffer[2 * m], sizeof(float) * m);
        cg::memcpy_async(tb, shx, &pbuffer[3 * m], sizeof(float) * m);
        cg::wait(tb);  // following `cr_thread_block()` needs sh*
#endif

        cr_thread_block(tb, sha, shc, shrhs, shx, m);

#ifdef EXPERIMENTAL_ASYNC_COPY
        if (idx < m) {
            memcpy_async(sha[idx - params.st], eq.a[idx], pipe);
            memcpy_async(shc[idx - params.st], eq.c[idx], pipe);
            memcpy_async(shrhs[idx - params.st], eq.rhs[idx], pipe);
        }
#else
        // we only modified first `m` elements.
        cg::memcpy_async(tb, sha, &eq.a[params.st], sizeof(float) * m);
        cg::memcpy_async(tb, shc, &eq.c[params.st], sizeof(float) * m);
        cg::memcpy_async(tb, shrhs, &eq.rhs[params.st], sizeof(float) * m);
#endif
        // copy back
        if (idx < m) {
            int dst = (idx + 1) * s - 1;
            assert(dst < params.n);
            eq.x[dst] = shx[idx];
        }
    }
}

/**
 * @brief         TPR Stage 3
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory, `eq.x` should be address in GLOBAL
 *                        memory
 * @param[in]     params  The parameters of TPR
 */
__device__ void TPR_CU::tpr_st3_ker(cg::thread_block &tb, Equation &eq,
                                    TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int i = tb.thread_index().x;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));
    assert(__isGlobal((void *)eq.x));

    for (int p = static_cast<int>(log2f(static_cast<float>(params.s))) - 1;
         p >= 0; p--) {
        int u = 1 << p;

        if (idx < params.n && ((idx - params.st - u + 1) % (2 * u) == 0) &&
            ((idx - params.st - u + 1) >= 0)) {
            int lidx = idx - u;
            float x_u;
            if (lidx < 0) {
                x_u = 0.0f;
            } else {
                x_u = eq.x[lidx];
            }

            eq.x[idx] = eq.rhs[i] - eq.a[i] * x_u - eq.c[i] * eq.x[idx + u];
        }
        tb.sync();
    }
    return;
}

/**
 * @brief      CR
 *
 *             Solve A*x = B by CR, where A is an n-by-n tridiagonal matrix, B
 *             is the right-hand-side vector of legth n
 *
 * @note       Only works in a block.
 *
 * @param      tb    cg::thread_block
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A. `n` should be power of 2
 */
__global__ void TPR_CU::cr_ker(float *a, float *c, float *rhs, float *x,
                               int n) {
    cg::thread_block tb = cg::this_thread_block();
    cr_thread_block(tb, a, c, rhs, x, n);
}

/**
 * @brief      CR
 *
 *             Solve A*x = B by CR, where A is an n-by-n tridiagonal matrix, B
 *             is the right-hand-side vector of legth n
 *
 * @note       Only works in a block.
 *
 * @param      tb    cg::thread_block
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A. `n` should be power of 2
 */
__device__ void TPR_CU::cr_thread_block(cg::thread_block &tb, float *a,
                                        float *c, float *rhs, float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int p = 0; p < static_cast<int>(log2f(static_cast<float>(n))) - 1;
         p++) {
        float tmp_aa, tmp_cc, tmp_rr;
        int u = 1 << p;  // offset
        int ux = 1 << (p + 1);
        bool condition =
            (idx < n) && ((idx - ux + 1) % ux == 0) && ((idx - ux + 1) >= 0);

        // reduction
        if (condition) {
            float akl, ckl, rkl;
            {
                int lidx = idx - u;
                if (lidx < 0) {
                    akl = -1.0f;
                    ckl = 0.0f;
                    rkl = 0.0f;
                } else {
                    akl = a[lidx];
                    ckl = c[lidx];
                    rkl = rhs[lidx];
                }
            }
            float akr, ckr, rkr;
            {
                int ridx = idx + u;
                if (ridx >= n) {
                    akr = 0.0f;
                    ckr = -1.0f;
                    rkr = 0.0f;
                } else {
                    akr = a[ridx];
                    ckr = c[ridx];
                    rkr = rhs[ridx];
                }
            }

            float inv_diag_k = 1.0f / (1.0f - ckl * a[idx] - akr * c[idx]);

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
        float inv_det = 1.0f / (1.0f - c[idx] * a[idx + u]);

        x[idx] = (rhs[idx] - c[idx] * rhs[idx + u]) * inv_det;
        x[idx + u] = (rhs[idx + u] - rhs[idx] * a[idx + u]) * inv_det;
    } else if (n == 1) {
        x[0] = rhs[0];
    }

    __syncthreads();

    for (int p = static_cast<int>(log2f(static_cast<float>(n))) - 2; p >= 0;
         p--) {
        int u = 1 << p;
        int ux = 1 << (p + 1);

        if ((idx < n) && ((idx - u + 1) % ux == 0) && (idx - u + 1 >= 0)) {
            int lidx = idx - u;
            float x_u;
            if (lidx < 0) {
                x_u = 0.0f;
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

/**
 * @brief      Helper function for tpr_cu
 *
 *             Solve A*x = B by TPR, where A is an n-by-n tridiagonal matrix, B
 *             is the right-hand-side vector of legth n
 *
 * @note       assert the diagonal elements of A are 1.0
 *
 * 1. check if device support cooperative launch
 * 2. allocate device memory for compute
 * 3. launch kernel `TPR_CU::tpr_ker`
 * 4. copy the answer from device to host
 * 5. free device memory
 *
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A. `n` should be power of 2
 * @param[in]  s     The parameter of TPR. Each block handles `s` equations.
 */
void TPR_CU::tpr_cu(float *a, float *c, float *rhs, float *x, int n, int s) {
    if (n / s > s) {
        fprintf(stderr, "Not supported parameters given. (n, s)=(%d, %d)\n", n,
                s);
        return;
    }
    int dev = 0;
    int size = n * sizeof(float);

    // Device
    float *d_a, *d_c, *d_r;  // device copies of a, c, rhs
    float *d_x, *d_pbuffer;
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));
    CU_CHECK(cudaMalloc((void **)&d_x, size));
    CU_CHECK(cudaMalloc((void **)&d_pbuffer, 4 * n / s * sizeof(float)));

    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    // launch configuration
    void *kernel_args[] = {&d_a, &d_c, &d_r, &d_x, &d_pbuffer, &n, &s};
    auto config = tpr_launch_config(n, s, dev);
    // auto [dim_grid, dim_block, shmem_size] = rhs; not supported
    auto dim_grid = std::get<0>(config);
    auto dim_block = std::get<1>(config);
    auto shmem_size = std::get<2>(config);

#ifdef TPR_PERF
    {
        time_ms elapsed = 0;
        pmcpp::DeviceTimer timer;
        timer.start();
#endif
        // launch
        CU_CHECK(cudaLaunchCooperativeKernel(
            (void *)tpr_ker, dim_grid, dim_block, kernel_args, shmem_size));
#ifdef TPR_PERF
        timer.stop_and_elapsed(elapsed);  // cudaDeviceSynchronize called
        pmcpp::perf_time.push_back(elapsed);
    }
#else
    cudaDeviceSynchronize();
#endif

    CU_CHECK(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));

    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    CU_CHECK(cudaFree(d_x));
    CU_CHECK(cudaFree(d_pbuffer));
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

#ifdef _REAL_IS_DOUBLE_
void TPR_CU::tpr_cu(double *a, double *c, double *rhs, double *x, int n,
                    int s) {
#pragma message " NOT IMPLEMENT."
}

void TPR_CU::cr_cu(double *a, double *c, double *rhs, double *x, int n) {
#pragma message " NOT IMPLEMENT."
}
#endif
