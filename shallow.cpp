#include "cuda_context.h"
#include "mem.h"
#include "unknowns.h"

#include <iostream>

namespace gpu = cuda_helper;

template<typename real, int time_order>
class Solver {
    const int M, N;
    const double C;
    const double xlen, ylen;
    const double hx, hy;

    typedef mem::template gpu_array<sloped<real> > gpu_sloped;
    typedef mem::template host_array<sloped<real> > host_sloped;

    typedef unknowns<real, gpu::allocator<sloped<real> > > gpu_unknowns;
    typedef unknowns<real> host_unknowns;

    gpu_unknowns u;
    gpu_unknowns v;
    gpu_unknowns fx;
    gpu_unknowns fy;
    gpu_sloped b;

    host_unknowns u_host;
    host_sloped b_host;

    double t, dt;

public:

    Solver(const int M, const int N, const double xlen, const double ylen, const double C)
        : M(M), N(N), C(C), xlen(xlen), ylen(ylen), hx(xlen / M), hy(ylen / N),
        u(M+2, N+2), v(M+2, N+2), fx(M+1, N), fy(M, N+1), b(M+2, N+2),
        u_host(M+2, N+2), b_host(M+2, N+2)
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
        throw "not implemented";
    }

    void add_fluxes_and_rhs(const double dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        throw "not implemented";
    }

    void limit_slopes(gpu_unknowns &u) {
        throw "not implemented";
    }

    void euler_limited(const real dt, const gpu_unknowns &u0, gpu_unknowns &u) {
        compute_fluxes(u0);
        add_fluxes_and_rhs(dt, u0, u);
        limit_slopes(u);
    }

public:

    void step() {
        if (time_order == 1) {
            euler_limited(dt, u, u);
        }
        if (time_order == 2) {
            euler_limited(dt, u, v);
            euler_limited(dt, v, v);
            u.blend_with(1. / 2, v);
        }
        if (time_order == 3) {
            euler_limited(dt, u, v);
            euler_limited(dt, v, v);
            v.blend_with(1. / 4, u);
            euler_limited(dt, v, v);
            u.blend_with(2. / 3, v);
        }

        t += dt;
    }

    double time() const { return t; }

    void save() {
        u_host = u;
        mem::copy(b_host, b);
    }
};

int main() {
    gpu::cuda_context cc;
    Solver<float, 2> s(50, 100, 1, 1, .15);

    s.save();
    s.step();

    return 0;
}
