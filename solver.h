#ifndef __SOLVER_H__
#define __SOLVER_H__

#define NOT_IMPLEMENTED throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " error: " + __func__ + " not implemented");

#include "solver_context.h"
#include "mem.h"
#include "unknowns.h"
#include "sloped.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>

template<typename real, int time_order, class Problem>
class Solver {
    const size_t M, N;
    const real C;
    const real hx, hy;

    typedef mem::template  gpu_array<real>           gpu_array;
    typedef mem::template  gpu_array<sloped<real> >  gpu_sloped_array;
    typedef mem::template host_array<sloped<real> > host_sloped_array;

    typedef unknowns< gpu_array>         gpu_flux;
    typedef unknowns< gpu_sloped_array>  gpu_unknowns;
    typedef unknowns<host_sloped_array> host_unknowns;

    gpu_unknowns u;
    gpu_unknowns v;
    gpu_flux fx;
    gpu_flux fy;
    gpu_sloped_array b;

    host_unknowns u_host;
    host_sloped_array b_host;

    std::shared_ptr<solver_context<real> > sctx;

    real t, dt;
    int _step;

    Problem prob;

public:

    Solver(const int M, const int N, const real C,
            const Problem &prob,
            std::shared_ptr<solver_context<real> > ctx
    ) :
        M(M), N(N), C(C), hx(prob.xlen() / M), hy(prob.ylen() / N),
        u(M+2, N+2), v(M+2, N+2), fx(M+1, N), fy(M+2, N+1), b(M+2, N+2),
        u_host(M+2, N+2), b_host(M+2, N+2), sctx(std::move(ctx)), prob(prob)
    {
        if (time_order == 1)
            std::cerr << "Method is likely to be unstable for any C" << std::endl;
        if (time_order == 2 && C > 1. / 3)
            std::cerr << "Method is likely to be unstable for C > 1/3" << std::endl;
        if (time_order == 3 && C > .409)
            std::cerr << "Method is likely to be unstable for C > .409" << std::endl;

        initialize();
    }

private:

    void initialize() {
        t = 0;
        _step = 0;

        for (size_t i = 0; i <= M + 1; i++)
            for (size_t j = 0; j <= N + 1; j++) {
                real x = (i - 1) * hx;
                real y = (j - 1) * hy;

                prob.initial(x, y, b_host(i, j),
                        u_host.h(i, j), u_host.hu(i, j), u_host.hv(i, j));
            }
        u = u_host;
        b = b_host;

//        sctx->deriv_to_slope(hx, hy, b, u);
        sctx->deriv_to_slope(0, 0, b, u);
    }

    void compute_fluxes(const gpu_unknowns &u) {
        sctx->compute_fluxes(u, fx, fy);
    }

    void add_fluxes_and_rhs(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        sctx->add_fluxes_and_rhs(dt, u0, b, fx, fy, u);
    }

    void limit_slopes(gpu_unknowns &u) {
//        NOT_IMPLEMENTED;
    }

    void euler_limited(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        compute_fluxes(u0);
        add_fluxes_and_rhs(dt, u0, u);
        limit_slopes(u);
    }

    void blend_with(gpu_unknowns &target, const real weight, const gpu_unknowns &source) {
        sctx->blend(target, weight, source);
    }

public:

    int step() { return _step; }

    void perform_step() {
        if (time_order == 1) {
            euler_limited(dt, u, u);
        }
        if (time_order == 2) {
            euler_limited(dt, u, v);
            euler_limited(dt, v, v);
            blend_with(u, 1. / 2, v);
        }
        if (time_order == 3) {
            euler_limited(dt, u, v);
            euler_limited(dt, v, v);
            blend_with(v, 1. / 4, u);
            euler_limited(dt, v, v);
            blend_with(u, 2. / 3, v);
        }

        t += dt;
        _step++;
    }

    real time() const { return t; }

    void save(const std::string &prefix);
};

#include "vtk.cpp"

#endif
