#ifndef __SOLVER_CONTEXT_H__
#define __SOLVER_CONTEXT_H__

#include "cuda_context.h"
#include "unknowns.h"

template<typename real>
struct type_prefix {
    static const std::string get();
};

template<> struct type_prefix<float> {
    static const std::string get() { return std::string("s"); }
};

template<> struct type_prefix<double> {
    static const std::string get() { return std::string("d"); }
};

#define DECLARE_PREFIXED_KERNEL(T, name, classname, member) class __kernel_ ## name { mutable CUfunction __f; const char *__name; \
    public: __kernel_ ## name() : __f(0), __name(#name) { } \
    cuda_helper::configured_call operator()(cuda_helper::dim3 grid, cuda_helper::dim3 block, unsigned int shmem = 0, CUstream stream = 0) const { \
        if (!__f) { __f = CUDA_HELPER_OUTERCLASS(classname, member)->lookup((type_prefix<T>::get() + __name).c_str()); } \
        return cuda_helper::configured_call(__f, grid, block, shmem, stream); \
    } } member

template<size_t align>
constexpr size_t ceildiv(size_t n) {
    static_assert(!(align & (align - 1)), "ailgn size must be a power of two");
    return (n + align - 1) / align;
}

template<typename real>
struct solver_context : public cuda_helper::cuda_context {
    typedef unknowns<real, cuda_helper::allocator<sloped<real> > > gpu_unknowns;

    DECLARE_PREFIXED_KERNEL(real, blend, solver_context, GPU_blend);

    solver_context(const int devid = 0, unsigned int flags = CU_CTX_SCHED_AUTO, bool performInit = true)
        : cuda_context(devid, flags, performInit)
    { }

    void blend(gpu_unknowns &u, const real w, const gpu_unknowns &o)
    {
        size_t m = u.h.m();
        size_t n = u.h.n();
        size_t ld = u.h.ld();

        const sloped<real> *oh  = o.h.data();
        const sloped<real> *ohu = o.hu.data();
        const sloped<real> *ohv = o.hv.data();

        sloped<real> *uh  = u.h.data();
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
};

#endif
