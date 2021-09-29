#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <array>
#include <tuple>
#include <vector>

#define EPS 1e-3

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
    int idx;
    int st;
    int ed;
};

// for main function use
class TPR_ANS {
   public:
    int n;
    int s;
    float *x;
    std::vector<float> data;

    TPR_ANS(int n) {
        this->n = n;
        this->data.resize(n);
        this->x = this->data.data();
    }

    ~TPR_ANS() {}

    bool operator==(const TPR_ANS &ans) {
        assert(this->n == ans.n);
        bool ret = true;

        for (int i = 0; i < this->n; i++) {
            ret &= fequal(this->x[i], ans.x[i]);
        }

        return true;
    }

    // operator `<<` cannot define for following reason,
    // nvcc 11: Compile Fail
    // `error: too many parameters for this operator function`
    // std::ostream& operator<<(std::ostream &os, TPR_ANS const &ans) {
    //     for (int i = 0; i < n; i++) {
    //         os << ans.x[i] << ", ";
    //     }
    //     return os << "\n";
    // }
    void display(std::ostream &os) {
        for (int i = 0; i < n; i++) {
            os << this->x[i] << ", ";
        }
        os << "\n";
    }

   private:
    bool fequal(float a, float b) {
        return fabs(a - b) <= EPS * fmax(1, fmax(fabs(a), fabs(b)));
    }
};

__global__ void tpr_ker(float *a, float *b, float *c, float *x, int n, int s);
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

#undef EPS
