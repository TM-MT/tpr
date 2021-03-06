#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <iostream>

#include "lib.hpp"
#include "main.hpp"
#include "ptpr.cuh"

#ifdef TPR_PERF
#include "pm.cuh"
#endif

/**
 * @brief Use `nvcuda::experimental` futures for cuda < 11.4
 */
#if (__CUDACC_VER_MAJOR__ <= 11) && (__CUDACC_VER_MINOR__ < 4)
#pragma message("Using Experimental Features")
#define EXPERIMENTAL_ASYNC_COPY
#endif

namespace cg = cooperative_groups;
#ifdef EXPERIMENTAL_ASYNC_COPY
using namespace nvcuda::experimental;
#endif

using namespace PTPR_CU;

/**
 * for dynamic shared memory use
 */
extern __shared__ float array[];

/**
 * @brief      PTPR main kernel
 *
 * @param[in]  a        a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c        c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                      0.0
 * @param[in]  rhs      rhs[0:n] The right-hand-side of the equation.
 * @param[out] x        x[0:n] for the solution
 * @param      pbuffer  Additional memory for Stage 2 use. pbuffer[0:3 * n / s]
 * @param[in]  n        The order of A. `n` should be power of 2
 * @param[in]  s        The parameter of PCR-like TPR. Each block handles `s`
 *                      equations.
 */
__global__ void PTPR_CU::tpr_ker(float *a, float *c, float *rhs, float *x,
                                 float *pbuffer, int n, int s) {
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
    params.ed = ed;

    // bkups, .x -> a, .y -> c, .z -> rhs
    float3 bkup_st;

#ifdef EXPERIMENTAL_ASYNC_COPY
    pipe.commit_and_wait();
#else
    cg::wait(tb);
#endif
    tpr_st1_ker(tb, eq, params);

    tpr_inter(tb, eq, bkup_st, params);

    tb.sync();

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

    tpr_st2_ker(tg, tb, eq, params, pbuffer);

    tg.sync();

    // from st2_ker
#ifdef EXPERIMENTAL_ASYNC_COPY
    if (blockIdx.x == 0 && idx < params.m) {
        pipe.commit_and_wait();
    }
#else
    if (blockIdx.x == 0) {
        cg::wait(tb);
    }
#endif

    // stage 3
    // assert sh* has data
    if (idx < n && idx == st) {
        // idx - st == 0
        sha[idx - st] = bkup_st.x;
        shc[idx - st] = bkup_st.y;
        shrhs[idx - st] = bkup_st.z;
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
 * @brief         PTPR Stage 1
 *
 *                PCR-like TPR Stage 1 On each block,
 * 1. apply PCR
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory
 * @param[in]     params  The parameters of PTPR
 */
__device__ void PTPR_CU::tpr_st1_ker(cg::thread_block &tb, Equation eq,
                                     TPR_Params const &params) {
    int idx = params.idx;
    int i = tb.thread_index().x;
    int n = params.n, s = params.s;
    float tmp_aa, tmp_cc, tmp_rr;
    float *sha = eq.a, *shc = eq.c, *shrhs = eq.rhs;
    assert(__isShared((void *)sha));
    assert(__isShared((void *)shc));
    assert(__isShared((void *)shrhs));

    for (int p = 1; p <= static_cast<int>(log2f(static_cast<float>(s))); p++) {
        if (idx < n) {
            // reduction
            int u = 1 << (p - 1);  // offset
            int lidx = i - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0f;
                ckl = 0.0f;
                rkl = 0.0f;
            } else {
                akl = sha[lidx];
                ckl = shc[lidx];
                rkl = shrhs[lidx];
            }
            int ridx = i + u;
            float akr, ckr, rkr;
            if (ridx >= s) {
                akr = 0.0f;
                ckr = -1.0f;
                rkr = 0.0f;
            } else {
                akr = sha[ridx];
                ckr = shc[ridx];
                rkr = shrhs[ridx];
            }

            float inv_diag_k = 1.0f / (1.0f - ckl * sha[i] - akr * shc[i]);

            tmp_aa = -inv_diag_k * akl * sha[i];
            tmp_cc = -inv_diag_k * ckr * shc[i];
            tmp_rr = inv_diag_k * (shrhs[i] - rkl * sha[i] - rkr * shc[i]);
        }

        tb.sync();

        if (idx < n) {
            // copy back
            sha[i] = tmp_aa;
            shc[i] = tmp_cc;
            shrhs[i] = tmp_rr;
        }

        tb.sync();
    }
}

/**
 * @brief         PTPR Intermediate stage 1
 *
 *                Update E_{st} by E_{ed}
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory
 * @param[out]    bkup    The bkup for stage 3 use. bkup->x: a, bkup->y: c,
 *                        bkup->z: rhs
 * @param[in]     params  The parameters of PTPR
 */
__device__ void PTPR_CU::tpr_inter(cg::thread_block &tb, Equation eq,
                                   float3 &bkup, TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;

    if ((idx < params.n) && (idx == params.st)) {
        int st = idx - params.st;  // == 0,
        /**
        FIXME: writing 0 cause compile error
        nvcc: V11.4.48, cuda: 11.4
        ```
        Invalid bitcast
        float* bitcast ([0 x float] addrspace(3)* @array to float*)
        Error: Broken function found, compilation aborted!
        ```
        **/
        int ed = params.s - 1;
        float s1;
        if (fabs(eq.c[ed]) < machine_epsilon) {
            s1 = -1.0;
        } else {
            s1 = -eq.c[st] / eq.c[ed];
        }

        bkup.x = eq.a[st];
        bkup.y = eq.c[st];
        bkup.z = eq.rhs[st];

        eq.a[st] += s1 * eq.a[ed];
        eq.c[st] = s1;
        eq.rhs[st] += s1 * eq.rhs[ed];
    }
}

/**
 * @brief      PTPR Intermediate stage GLOBAL
 *
 *             Update E_{st-1} by E_{st}
 *
 * @param      tb       cg::thread_block
 * @param[in]  eq       Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                      GLOBAL memory
 * @param[in]  params   The parameters of PTPR
 * @param[out] pbuffer  The pbuffer
 */
__device__ void PTPR_CU::tpr_inter_global(cg::thread_block &tb, Equation eq,
                                          TPR_Params const &params,
                                          float *pbuffer) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int ed = params.ed;
    int dst = idx / params.s;

    if (idx < params.n - 1 && idx == ed) {
        int k = idx, kr = idx + 1;  // (k, kr) = (st-1, st)
        float ak = eq.a[k], akr = eq.a[kr];
        float ck = eq.c[k], ckr = eq.c[kr];
        float rhsk = eq.rhs[k], rhskr = eq.rhs[kr];
        float inv_diag_k = 1.0f / (1.0f - akr * ck);

        pbuffer[dst] = inv_diag_k * ak;                    // a[k]
        pbuffer[params.m + dst] = -inv_diag_k * ckr * ck;  // c[k]
        pbuffer[2 * params.m + dst] =
            inv_diag_k * (rhsk - rhskr * ck);  // rhs[k]
    } else if (idx == params.n - 1) {
        pbuffer[params.m - 1] = eq.a[idx];
        pbuffer[2 * params.m - 1] = eq.c[idx];
        pbuffer[3 * params.m - 1] = eq.rhs[idx];
    }
}

/**
 * @brief         TPR Stage 2
 *
 *                PCR-like TPR Stage 2
 *                1. call `tpr_inter_gloabl`
 *                2. set up for PCR call
 *                3. call PCR
 *                4. copy back the solution
 *
 * @param         tg       cg::grid_group
 * @param         tb       cg::thread_block
 * @param[in,out] eq       Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                         GLOBAL memory, length of `n`
 * @param[in]     params   The parameters of PTPR
 * @param         pbuffer  The pbuffer
 */
__device__ void PTPR_CU::tpr_st2_ker(cg::grid_group &tg, cg::thread_block &tb,
                                     Equation eq, TPR_Params const &params,
                                     float *pbuffer) {
    tpr_inter_global(tb, eq, params, pbuffer);

    tg.sync();

    if (blockIdx.x == 0) {
        int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
        int m = params.m;
        int s = params.s;
        assert(m <= s);

        __shared__ float *sha, *shc, *shrhs;
        sha = (float *)array;
        shc = (float *)&array[s];
        shrhs = (float *)&array[2 * s];

#ifdef EXPERIMENTAL_ASYNC_COPY
        pipeline pipe;
        if (idx < m) {
            memcpy_async(sha[idx - params.st], pbuffer[idx], pipe);
            memcpy_async(shc[idx - params.st], pbuffer[m + idx], pipe);
            memcpy_async(shrhs[idx - params.st], pbuffer[2 * m + idx], pipe);
            pipe.commit_and_wait();
        }
#else
        cg::memcpy_async(tb, sha, &pbuffer[0], sizeof(float) * m);
        cg::memcpy_async(tb, shc, &pbuffer[m], sizeof(float) * m);
        cg::memcpy_async(tb, shrhs, &pbuffer[2 * m], sizeof(float) * m);
        cg::wait(tb);  // following `pcr_thread_block()` needs sh*
#endif

        pcr_thread_block(tb, sha, shc, shrhs, m);

        // copy back data in shared memory
        // `shrhs` has the answer, so one should not overwrite `shrhs`
#ifdef EXPERIMENTAL_ASYNC_COPY
        if (idx < m) {
            memcpy_async(sha[idx - params.st], eq.a[idx], pipe);
            memcpy_async(shc[idx - params.st], eq.c[idx], pipe);
        }
#else
        // we only modified first `m` elements.
        cg::memcpy_async(tb, sha, &eq.a[params.st], sizeof(float) * m);
        cg::memcpy_async(tb, shc, &eq.c[params.st], sizeof(float) * m);
#endif

        if (idx < m) {
            int dst = (idx + 1) * s - 1;
            assert(dst < params.n);
            eq.x[dst] = shrhs[idx];
        }

#ifdef EXPERIMENTAL_ASYNC_COPY
        if (idx < m) {
            memcpy_async(shrhs[idx - params.st], eq.rhs[idx], pipe);
        }
#else
        cg::memcpy_async(tb, shrhs, &eq.rhs[params.st], sizeof(float) * m);
#endif
    }
}

/**
 * @brief         PTPR Stage 3
 *
 * @param         tb      cg::thread_block
 * @param[in,out] eq      Equation. `eq.a, eq.c, eq.rhs` should be address in
 *                        shared memory
 * @param[in]     params  The parameters of PTPR
 */
__device__ void PTPR_CU::tpr_st3_ker(cg::thread_block &tb, Equation eq,
                                     TPR_Params const &params) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    int i = tb.thread_index().x;
    int s = params.s;
    assert(__isShared((void *)eq.a));
    assert(__isShared((void *)eq.c));
    assert(__isShared((void *)eq.rhs));
    assert(__isGlobal((void *)eq.x));

    if (idx < params.n) {
        int lidx = max(0, params.st - 1);

        float key =
            1.0f / eq.c[s - 1] *
            (eq.rhs[s - 1] - eq.a[s - 1] * eq.x[lidx] - eq.x[params.ed]);
        if (eq.c[s - 1] == 0.0f) {
            key = 0.0f;
        }

        eq.x[idx] = eq.rhs[i] - eq.a[i] * eq.x[lidx] - eq.c[i] * key;
    }
    return;
}

/**
 * @brief         PCR
 *
 *                Solve A*x = B, where A is an n-by-n tridiagonal matrix, B is
 *                the right-hand-side vector of legth n
 *
 * @note          assert the diagonal elements of A are 1.0
 * @note          Currently only works in a block.
 *
 * @param[in]     a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]     c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                      0.0
 * @param[in]     rhs   rhs[0:n] The right-hand-side of the equation. On exit,
 *                      it holds the solution.
 * @param[in,out] n     The order of A. `n` should be power of 2
 * @param      tb    cg::thread_block
 */
__global__ void PTPR_CU::pcr_ker(float *a, float *c, float *rhs, int n) {
    cg::thread_block tb = cg::this_thread_block();
    pcr_thread_block(tb, a, c, rhs, n);
}

/**
 * @brief         PCR
 * @note          assert the diagonal elements of A are 1.0
 * @note          Only works in a block.
 *
 * @param         tb    cg::thread_block
 * @param[in]     a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]     c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                      0.0
 * @param[in,out] rhs   rhs[0:n] The right-hand-side of the equation. On exit,
 *                      it holds the solution.
 * @param[in]     n     The order of A. `n` should be power of 2
 */
__device__ void PTPR_CU::pcr_thread_block(cg::thread_block &tb, float *a,
                                          float *c, float *rhs, int n) {
    int idx = tb.group_index().x * tb.group_dim().x + tb.thread_index().x;
    float tmp_aa, tmp_cc, tmp_rr;

    for (int p = 1; p <= static_cast<int>(log2f(static_cast<float>(n))); p++) {
        if (idx < n) {
            // reduction
            int u = 1 << (p - 1);  // offset
            int lidx = idx - u;
            float akl, ckl, rkl;
            if (lidx < 0) {
                akl = -1.0f;
                ckl = 0.0f;
                rkl = 0.0f;
            } else {
                akl = a[lidx];
                ckl = c[lidx];
                rkl = rhs[lidx];
            }
            int ridx = idx + u;
            float akr, ckr, rkr;
            if (ridx >= n) {
                akr = 0.0f;
                ckr = -1.0f;
                rkr = 0.0f;
            } else {
                akr = a[ridx];
                ckr = c[ridx];
                rkr = rhs[ridx];
            }

            float inv_diag_k = 1.0f / (1.0f - ckl * a[idx] - akr * c[idx]);

            tmp_aa = -inv_diag_k * akl * a[idx];
            tmp_cc = -inv_diag_k * ckr * c[idx];
            tmp_rr = inv_diag_k * (rhs[idx] - rkl * a[idx] - rkr * c[idx]);
        }

        tb.sync();

        if (idx < n) {
            // copy back
            a[idx] = tmp_aa;
            c[idx] = tmp_cc;
            rhs[idx] = tmp_rr;
        }

        tb.sync();
    }
}

/**
 * @brief      Check return value of `expr`
 *
 * @param      expr  The expression
 *
 * @return     Nothing if success else exit
 */
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
 * @brief      Helper function for ptpr_cu
 *
 *             Solve A*x = B, where A is an n-by-n tridiagonal matrix, B is the
 *             right-hand-side vector of legth n
 *
 * @note       assert the diagonal elements of A are 1.0
 *
 * 1. check if device support cooperative launch
 * 2. allocate device memory for compute
 * 3. launch kernel `PTPR_CU::tpr_cu`
 * 4. copy the answer from device to host
 * 5. free device memory
 *
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A. `n` should be power of 2
 * @param[in]  s     The parameter of PCR-like TPR. Each block handles `s`
 *                   equations.
 */
void PTPR_CU::ptpr_cu(float *a, float *c, float *rhs, float *x, int n, int s) {
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
    CU_CHECK(cudaMalloc((void **)&d_pbuffer, 3 * n / s * sizeof(float)));

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
 * @brief      launch configuration for tpr_ker
 * @details    calculate suitable dimension and shared memory size for tpr_ker
 *
 * @param[in]  n     size of the equation
 * @param[in]  s     TPR parameter
 * @param[in]  dev   cuda device id
 *
 * @return     [dim_grid, dim_block, shared_memory_size]
 */
std::tuple<dim3, dim3, size_t> PTPR_CU::tpr_launch_config(int n, int s,
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
 * @brief      Helper function for tpr_cu
 * @details    calculate dimension for cuda kernel launch.
 *
 * @param[in]  n     size of the equation
 * @param[in]  s     TPR parameter
 * @param[in]  dev   cuda device id
 *
 * @return     [dim_grid, dim_block]
 */
std::array<dim3, 2> PTPR_CU::n2dim(int n, int s, int dev) {
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

/**
 * @brief      Helper function for pcr_ker
 *
 *             Solve A*x = B by PCR, where A is an n-by-n tridiagonal matrix, B
 * is the right-hand-side vector of legth n
 *
 * @note       assert the diagonal elements of A are 1.0
 *
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A.
 */
void PTPR_CU::pcr_cu(float *a, float *c, float *rhs, float *x, int n) {
    int size = n * sizeof(float);

    // Device
    float *d_a, *d_c, *d_r;  // device copies of a, c, rhs
    CU_CHECK(cudaMalloc((void **)&d_a, size));
    CU_CHECK(cudaMalloc((void **)&d_c, size));
    CU_CHECK(cudaMalloc((void **)&d_r, size));

    CU_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_r, rhs, size, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    pcr_ker<<<1, n>>>(d_a, d_c, d_r, n);

    cudaDeviceSynchronize();
    CU_CHECK(cudaMemcpy(x, d_r, size, cudaMemcpyDeviceToHost));

    CU_CHECK(cudaFree(d_a));
    CU_CHECK(cudaFree(d_c));
    CU_CHECK(cudaFree(d_r));
    return;
}

#ifdef _REAL_IS_DOUBLE_
void PTPR_CU::ptpr_cu(double *a, double *c, double *rhs, double *x, int n,
                      int s) {
#pragma message " NOT IMPLEMENT."
}

void PTPR_CU::pcr_cu(double *a, double *c, double *rhs, double *x, int n) {
#pragma message " NOT IMPLEMENT."
}
#endif
