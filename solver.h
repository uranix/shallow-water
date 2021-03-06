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
    const size_t M, N, LD;
    const real C;
    const real hx, hy;

    typedef mem::template  gpu_array<real>           gpu_array;
    typedef mem::template host_array<real>          host_array;
    typedef mem::template  gpu_array<sloped<real> >  gpu_sloped_array;
    typedef mem::template host_array<sloped<real> > host_sloped_array;

    typedef unknowns< gpu_array>         gpu_flux;
    typedef unknowns<host_array>        host_flux;
    typedef unknowns< gpu_sloped_array>  gpu_unknowns;
    typedef unknowns<host_sloped_array> host_unknowns;

    gpu_unknowns u;
    gpu_unknowns v;
    gpu_flux fx;
    gpu_flux fy;
    gpu_sloped_array b;

    host_unknowns u_host;
    host_sloped_array b_host;
    host_flux fx_host;
    host_flux fy_host;

    std::shared_ptr<solver_context<real> > sctx;

    real t, dt;
    int _step;

    Problem prob;

public:

    Solver(const int M, const int N, const real C,
            const Problem &prob,
            std::shared_ptr<solver_context<real> > ctx
    ) :
        M(M), N(N), LD(alignup<16>(M+2)), C(C),
        hx(prob.xlen() / M), hy(prob.ylen() / N),
        u (M+2, N+2, LD),
        v (M+2, N+2, LD),
        fx(M+1, N  , LD),
        fy(M+2, N+1, LD),
        b (M+2, N+2, LD),
        u_host(M+2, N+2, LD),
        b_host(M+2, N+2, LD),
        fx_host(M+1, N  , LD),
        fy_host(M+2, N+1, LD),
        sctx(std::move(ctx)),
        prob(prob)
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

        for (size_t i = 1; i <= M; i++)
            for (size_t j = 1; j <= N; j++) {
                real x = (i - 0.5) * hx;
                real y = (j - 0.5) * hy;

                prob.initial(x, y, b_host(i, j),
                        u_host.h(i, j), u_host.hu(i, j), u_host.hv(i, j));
            }
        u = u_host;
        b = b_host;

        sctx->deriv_to_slope(hx, hy, b, u);
    }

    void dump(const host_flux &f, const char *name, size_t s = 0) {
        std::cout << name << ".h: " << std::endl;
        for (size_t j = 0; j < f.n(); j++) {
            for (size_t i = s; i < f.m() - s; i++)
                printf("%10.6f ", f.h(i, j));
            printf("\n");
        }
        std::cout << name << ".hu: " << std::endl;
        for (size_t j = 0; j < f.n(); j++) {
            for (size_t i = s; i < f.m() - s; i++)
                printf("%10.6f ", f.hu(i, j));
            printf("\n");
        }
        std::cout << name << ".hv: " << std::endl;
        for (size_t j = 0; j < f.n(); j++) {
            for (size_t i = s; i < f.m() - s; i++)
                printf("%10.6f ", f.hv(i, j));
            printf("\n");
        }
    }

    void compute_fluxes(const gpu_unknowns &u) {
        sctx->compute_fluxes(u, fx, fy);

/*        fx_host = fx;
        fy_host = fy;

        dump(fx_host, "fx");
        dump(fy_host, "fy", 1);*/
    }

    void add_fluxes_and_rhs(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        sctx->add_fluxes_and_rhs(dt, hx, hy, u0, b, fx, fy, u);
    }

    void limit_slopes(gpu_unknowns &u) {
        sctx->compute_slopes(b, u, fx, fy);
        sctx->limit_slopes(b, fx, fy, u);
    }

    void euler_limited(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        compute_fluxes(u0);
        add_fluxes_and_rhs(dt, u0, u);
        limit_slopes(u);
    }

    void blend_with(gpu_unknowns &target, const real weight, const gpu_unknowns &source) {
        sctx->blend(target, weight, source);
    }

    real estimate_timestep(const gpu_unknowns &u) {
        real cmax = sctx->compute_max_speed(u);
        return C * std::min(hx, hy) / cmax;
    }

public:

    real tmax() const { return prob.tmax(); }
    real next(real t) const { return prob.next(t); }
    int step() const { return _step; }

    void perform_step() {
        dt = estimate_timestep(u);
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
