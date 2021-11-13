#pragma once
#include "lib.hpp"

class PCR : Solver {
    /**
     * a[0:n+2*margin]
     * a: 0 ... 0 a[0] a[1] .... a[n-1] 0 ... 0
     *    <-----> <------elements-----> <----->
     *      |                             |
     *      \- extra length of `margin`   \-  extra
     */
    real *a, *c, *rhs;
    real *a1, *c1, *rhs1;
    int n;
    bool using_local_allocation =
        false;  // `a, c, rhs` are allocated by PCR when true

   public:
    int margin;

    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, diag, c, rhs);
    };

    PCR(){};

    ~PCR() {
        if (this->using_local_allocation) {
            free_extend_input_array(a, this->n);
            free_extend_input_array(c, this->n);
            free_extend_input_array(rhs, this->n);
        }
        delete[] this->a1;
        delete[] this->c1;
        delete[] this->rhs1;
    }

    PCR(const PCR &pcr) {
        n = pcr.n;
        init(pcr.n);
    };

    void init(int n) {
        this->n = n;
        this->margin = array_margin(n);

        this->a1 = new real[n];
        this->c1 = new real[n];
        this->rhs1 = new real[n];
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs) {
        this->using_local_allocation = true;
        this->a = extend_input_array(a, n);
        this->c = extend_input_array(c, n);
        this->rhs = extend_input_array(rhs, n);
    }

    /**
     * @brief      Sets the tridiagonal system.
     *
     * @param      a     a[0:n+2*margin], a[margin:margin+n] holds the
     *                   subdiagonal elements of A.
     * @param      diag  nullptr
     * @param      c     c[0:n+2*margin], c[margin:margin+n] holds the
     *                   superdiagonal elements of A.
     * @param      rhs   rhs[0:n+2*margin], rhs[margin:margin+n] holds the
     *                   right-hand-side of the equation.
     */
    void set_tridiagonal_system_no_extend(real *a, real *diag, real *c,
                                          real *rhs) {
        this->a = a;
        this->c = c;
        this->rhs = rhs;
    }

    int solve();

    int get_ans(real *x);

    real *create_extend_array(int oldlen);
    real *create_extend_array(int oldlen, int margin);
    real *extend_input_array(real *p, int len);
    void free_extend_input_array(real *p, int len);
    inline int array_margin(int n) { return 1 << fllog2(n); }
};
