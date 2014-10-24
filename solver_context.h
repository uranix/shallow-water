#ifndef __SOLVER_CONTEXT_H__
#define __SOLVER_CONTEXT_H__

#include "cuda_context.h"
#include "unknowns.h"
#include "sloped.h"

template<typename real>
struct type_suffix {
    static const std::string get();
};

template<> struct type_suffix<float> {
    static const std::string get() { return std::string("_float"); }
};

template<> struct type_suffix<double> {
    static const std::string get() { return std::string("_double"); }
};

#define DECLARE_PREFIXED_KERNEL(T, name, classname, member) class __kernel_ ## name { mutable CUfunction __f; const char *__name; \
    public: __kernel_ ## name() : __f(0), __name(#name) { } \
    cuda_helper::configured_call operator()(cuda_helper::dim3 grid, cuda_helper::dim3 block, unsigned int shmem = 0, CUstream stream = 0) const { \
        if (!__f) { __f = CUDA_HELPER_OUTERCLASS(classname, member)->lookup((std::string(__name) + type_suffix<real>::get()).c_str()); } \
        return cuda_helper::configured_call(__f, grid, block, shmem, stream); \
    } } member

template<size_t align>
constexpr size_t ceildiv(size_t n) {
    static_assert(!(align & (align - 1)), "ailgn size must be a power of two");
    return (n + align - 1) / align;
}

template<typename real>
struct solver_context : public cuda_helper::cuda_context {
    typedef array2d<sloped<real>, cuda_helper::allocator<sloped<real> > > gpu_sloped_array;
    typedef array2d<       real , cuda_helper::allocator<       real  > > gpu_array;
    typedef unknowns<gpu_sloped_array> gpu_unknowns;
    typedef unknowns<gpu_array>        gpu_flux;

    DECLARE_PREFIXED_KERNEL(real, blend, solver_context, GPU_blend);
    DECLARE_PREFIXED_KERNEL(real, der2slope, solver_context, GPU_der2slope);

    solver_context(const int devid = 0, unsigned int flags = CU_CTX_SCHED_AUTO, bool performInit = true)
        : cuda_context(devid, flags, performInit)
    {
        load_module("kernels.ptx");
    }

    void blend(gpu_unknowns &u, const real w, const gpu_unknowns &o)
    {
        size_t m = u.h.m();
        size_t n = u.h.n();
        size_t ld = u.h.ld();

        const sloped<real> *oh  = o.h .data();
        const sloped<real> *ohu = o.hu.data();
        const sloped<real> *ohv = o.hv.data();

        sloped<real> *uh  = u.h .data();
        sloped<real> *uhu = u.hu.data();
        sloped<real> *uhv = u.hv.data();

        cuda_helper::dim3 block(32, 16);
        cuda_helper::dim3 grid(ceildiv<32>(m), ceildiv<16>(n));

        GPU_blend(grid, block)({
                &m, &n, &ld, &w,
                &uh, &uhu, &uhv,
                &oh, &ohu, &ohv
            });
    }

    void deriv_to_slope(const real hx, const real hy, gpu_sloped_array &barr, gpu_unknowns &u) {
        size_t m = barr.m();
        size_t n = barr.n();
        size_t ld = barr.ld();

        sloped<real> *b  = barr.data();
        sloped<real> *h  = u.h .data();
        sloped<real> *hu = u.hu.data();
        sloped<real> *hv = u.hv.data();

        cuda_helper::dim3 block(32, 16);
        cuda_helper::dim3 grid(ceildiv<32>(m), ceildiv<16>(n));

        GPU_der2slope(grid, block)({
                &m, &n, &ld,
                &hx, &hy, &b,
                &h, &hu, &hv
            });
    }

    void compute_fluxes(const gpu_unknowns &u, gpu_flux &fx, gpu_flux &fy) {
        NOT_IMPLEMENTED;
    }
};

#endif
