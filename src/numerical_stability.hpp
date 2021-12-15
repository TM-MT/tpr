#pragma once
#include <glob.h>

#include <filesystem>
#include <string>
#include <vector>

#include "lib.hpp"
#include "system.hpp"

namespace nstab {
std::vector<std::string> glob(const std::string &pattern);
std::vector<std::string> get_input_files();
std::vector<real> read_xtrue(int n);
template <typename T>
double norm(T *solution, real *xtruth, int n);
double eval(Solver &solver, trisys::TRIDIAG_SYSTEM &sys,
            std::vector<real> &xtruth);
std::filesystem::path filename(std::string &path);
}  // namespace nstab
