#include "numerical_stability.hpp"

#include <glob.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cr.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "ptpr.hpp"
#include "system.hpp"
#include "tpr.hpp"

#ifdef INCLUDE_REFERENCE_LAPACK
#include "reference_lapack.hpp"
#endif

#ifdef BUILD_CUDA
#include "ptpr.cuh"
#include "tpr.cuh"
#endif

namespace nstab {
namespace fs = std::filesystem;

std::vector<std::string> glob(const std::string &pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

std::vector<std::string> get_input_files() {
    auto pattern = std::string(INPUTS_DIR "/eq*.txt");
    return nstab::glob(pattern);
}

std::vector<real> read_xtrue(int n) {
    std::vector<real> ret;
    ret.resize(n);
    auto path = std::string(INPUTS_DIR "/xt.txt");
    FILE *fp = fopen(path.c_str(), "r");

    if (fp != nullptr) {
        for (int i = 0; i < n; i++) {
            assert(fscanf(fp, "%f", &ret[i]) == 1);
        }
    } else {
        std::cerr << "Failed to read " << path << "\n";
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    return ret;
}

/**
 * @brief      L2 NORM
 *
 * @param      solution  The solution
 * @param      xtruth    The xtruth
 * @param[in]  n         { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
double norm(real *solution, real *xtruth, int n) {
    std::vector<double> diff_sq;
    diff_sq.resize(n);
    for (int i = 0; i < n; i++) {
        diff_sq[i] = (solution[i] - xtruth[i]) * (solution[i] - xtruth[i]);
    }
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum1 += diff_sq[i];
        sum2 += xtruth[i] * xtruth[i];
    }
    sum1 = sqrt(sum1);
    sum2 = sqrt(sum2);

    return sum1 / sum2;
}

template <typename T>
double eval(T &solver, trisys::TRIDIAG_SYSTEM &sys, std::vector<real> &xtruth) {
    solver.set_tridiagonal_system(sys.a, sys.diag, sys.c, sys.rhs);
    solver.solve();
    std::vector<real> solution;
    solution.resize(sys.n);
    solver.get_ans(&solution[0]);

    return norm(&solution[0], &xtruth[0], sys.n);
}

fs::path filename(std::string &path) { return fs::path(path).filename(); }
}  // namespace nstab

// for tpr, pcr-like tpr
#define TPR_EVAL_AND_PRINT(SOLVER, FILENAME, N, S)                     \
    {                                                                  \
        input.assign();                                                \
        SOLVER solver(N, S);                                           \
        double norm = nstab::eval(solver, input.sys, xt);              \
        printf("%s,%s,%d,%d,%le\n", nstab::filename(FILENAME).c_str(), \
               #SOLVER, N, S, norm);                                   \
    }

// For CR, PCR, Lapack etc.
#define EVAL_AND_PRINT(SOLVER, FILENAME, N)                            \
    {                                                                  \
        input.assign();                                                \
        SOLVER solver(N);                                              \
        double norm = nstab::eval(solver, input.sys, xt);              \
        printf("%s,%s,%d,%d,%le\n", nstab::filename(FILENAME).c_str(), \
               #SOLVER, N, 0, norm);                                   \
    }

int main() {
    int n = 512;

    auto xt = nstab::read_xtrue(n);

    for (auto &fname : nstab::get_input_files()) {
        trisys::ExampleFromInput input(n);
        input.path = fname;

        for (int s = 16; s <= n; s *= 2) {
            TPR_EVAL_AND_PRINT(PTPR, fname, n, s);
            TPR_EVAL_AND_PRINT(TPR, fname, n, s);
        }

        EVAL_AND_PRINT(PCR, fname, n);
        EVAL_AND_PRINT(REFERENCE_LAPACK, fname, n);
    }
}
