#pragma once
#include <lib.hpp>

class PCR: Solver
{
    real *a, *c, *rhs;
    int n;
    int margin;

public:
    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        this->a = extend_input_array(a, n);
        this->c = extend_input_array(c, n);
        this->rhs = extend_input_array(rhs, n);
        this->n = n;
        this->margin = array_margin(n);

        #pragma acc enter data create(this, this->n, this->margin)
        #pragma acc enter data copyin(this->a[-margin:n+margin], this->c[-margin:n+margin], this->rhs[-margin:n+margin]) wait
        // TO-DO
        // make sure diag = {1., 1., ..., 1.};
    };

    ~PCR() {
        #pragma acc exit data delete(this->a[-margin:n+margin], this->c[-margin:n+margin], this->rhs[-margin:n+margin])
        #pragma acc exit data delete(this, this->n, this->margin)
        delete &this->a[-this->margin];
        delete &this->c[-this->margin];
        delete &this->rhs[-this->margin];
    }
 
    int solve();

    int get_ans(real *x);

    real* extend_input_array(real *p, int len);

private:
    PCR(const PCR &pcr);
    PCR &operator=(const PCR &pcr);

    real* extend_array(real *p, int oldlen, int newlen, int margin);
    inline int array_margin(int n) {
        return 1 << fllog2(n);
    }
};
