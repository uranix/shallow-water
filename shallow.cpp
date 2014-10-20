#include "solver_context.h"
#include "mem.h"
#include "unknowns.h"

#include <iostream>
#include <memory>

namespace gpu = cuda_helper;

#define NOT_IMPLEMENTED throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " error: " + __func__ + " not implemented");

template<typename real, int time_order>
class Solver {
    const int M, N;
    const real C;
    const real xlen, ylen;
    const real hx, hy;

    typedef mem::template gpu_array<sloped<real> > gpu_sloped;
    typedef mem::template host_array<sloped<real> > host_sloped;

    typedef unknowns<real, gpu::allocator<sloped<real> > > gpu_unknowns;
    typedef unknowns<real> host_unknowns;

    gpu_unknowns u;
    gpu_unknowns v;
    gpu_unknowns fx;
    gpu_unknowns fy;
    gpu_sloped b;

    std::shared_ptr<solver_context<real> > sctx;

    host_unknowns u_host;
    host_sloped b_host;

    real t, dt;

public:

    Solver(const int M, const int N, const real xlen, const real ylen,
            const real C, std::shared_ptr<solver_context<real> > ctx
    ) :
        M(M), N(N), C(C), xlen(xlen), ylen(ylen), hx(xlen / M), hy(ylen / N),
        u(M+2, N+2), v(M+2, N+2), fx(M+1, N), fy(M, N+1), b(M+2, N+2),
        u_host(M+2, N+2), b_host(M+2, N+2), sctx(std::move(ctx))
    {
        if (time_order == 1)
            std::cerr << "Method is likely to be unstable for any C" << std::endl;
        if (time_order == 2 && C > 1. / 3)
            std::cerr << "Method is likely to be unstable for C > 1/3" << std::endl;
        if (time_order == 3 && C > .409)
            std::cerr << "Method is likely to be unstable for C > .409" << std::endl;

        t = 0;
    }

private:

    void compute_fluxes(const gpu_unknowns &u) {
        NOT_IMPLEMENTED;
    }

    void add_fluxes_and_rhs(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        NOT_IMPLEMENTED;
    }

    void limit_slopes(gpu_unknowns &u) {
        NOT_IMPLEMENTED;
    }

    void euler_limited(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        compute_fluxes(u0);
        add_fluxes_and_rhs(dt, u0, u);
        limit_slopes(u);
    }

    void blend_with(gpu_unknowns &target, const real weight, const gpu_unknowns &source) {
        NOT_IMPLEMENTED;
    }

public:

    void step() {
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
    }

    real time() const { return t; }

    void save() {
        u_host = u;
        mem::copy(b_host, b);
    }
};

int main() {
    try {
        auto ctx = std::make_shared<solver_context<float> >();
        Solver<float, 2> s(50, 100, 1, 1, .15, ctx);

        s.save();
        s.step();
    } catch (const char *msg) {
        std::cerr << "Exception msg = " << msg << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception msg = " << e.what() << std::endl;
    }
    return 0;
}
