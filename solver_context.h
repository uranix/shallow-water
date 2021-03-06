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

#define DECLARE_SUFFIXED_KERNEL(T, name, classname, member) class __kernel_ ## name { mutable CUfunction __f; const char *__name; \
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

    DECLARE_SUFFIXED_KERNEL(real, blend,        solver_context, GPU_blend);
    DECLARE_SUFFIXED_KERNEL(real, der2slope,    solver_context, GPU_der2slope);
    DECLARE_SUFFIXED_KERNEL(real, flux,         solver_context, GPU_fluxes);
    DECLARE_SUFFIXED_KERNEL(real, add_flux,     solver_context, GPU_add_flux);
    DECLARE_SUFFIXED_KERNEL(real, slope,        solver_context, GPU_slopes);
    DECLARE_SUFFIXED_KERNEL(real, limit_slopes, solver_context, GPU_limit_slopes);
    DECLARE_SUFFIXED_KERNEL(real, max_speed,    solver_context, GPU_max_speed);

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

        raw_unknowns<const sloped<real> *> oraw  = o.data();
        raw_unknowns<sloped<real> *> uraw = u.data();

        cuda_helper::dim3 block(32, 16);
        cuda_helper::dim3 grid(ceildiv<32>(m), ceildiv<16>(n));

        GPU_blend(grid, block)({
                &m, &n, &ld, &w,
                &uraw, &oraw
            });
    }

    void deriv_to_slope(const real hx, const real hy, gpu_sloped_array &b, gpu_unknowns &u) {
        size_t m  = b.m();
        size_t n  = b.n();
        size_t ld = b.ld();

        sloped<real> *braw  = b.data();
        raw_unknowns<sloped<real> *> uraw  = u.data();

        cuda_helper::dim3 block(32, 16);
        cuda_helper::dim3 grid(ceildiv<32>(m), ceildiv<16>(n));

        GPU_der2slope(grid, block)({
                &m, &n, &ld,
                &hx, &hy,
                &braw, &uraw
            });
    }

    void compute_fluxes(const gpu_unknowns &u, gpu_flux &fx, gpu_flux &fy) {
        size_t m  = u.m() - 2;
        size_t n  = u.n() - 2;
        size_t ld = u.ld();

        size_t lines = 32;

        raw_unknowns<const sloped<real> *> uraw = u.data();

        raw_unknowns<real *> fxraw  = fx.data();
        raw_unknowns<real *> fyraw  = fy.data();

        cuda_helper::dim3 block(128);
        size_t stride = block.x - 1;
        cuda_helper::dim3 grid((fx.m() + stride - 1) / stride, ceildiv<32>(n));

        GPU_fluxes(grid, block, sizeof(raw_unknowns<sloped<real> >) * block.x)({
                &m, &n, &ld, &lines, &stride,
                &uraw,
                &fxraw,
                &fyraw
            });
    }

    void compute_slopes(const gpu_sloped_array &b, const gpu_unknowns &u, gpu_flux &fx, gpu_flux &fy) {
        size_t m  = u.m() - 2;
        size_t n  = u.n() - 2;
        size_t ld = u.ld();

        size_t lines = 32;

        raw_unknowns<const sloped<real> *> uraw = u.data();

        raw_unknowns<real *> fxraw  = fx.data();
        raw_unknowns<real *> fyraw  = fy.data();

        const sloped<real> *braw = b.data();

        cuda_helper::dim3 block(128);
        size_t stride = block.x - 1;
        cuda_helper::dim3 grid((fx.m() + stride - 1) / stride, ceildiv<32>(n));

        GPU_slopes(grid, block, sizeof(raw_unknowns<sloped<real> >) * block.x)({
                &m, &n, &ld, &lines, &stride,
                &braw,
                &uraw,
                &fxraw,
                &fyraw
            });
    }

    void limit_slopes(const gpu_sloped_array &b, const gpu_flux &fx, const gpu_flux &fy, gpu_unknowns &u) {
        size_t m  = u.m() - 2;
        size_t n  = u.n() - 2;
        size_t ld = u.ld();

        size_t lines = 32;

        raw_unknowns<sloped<real> *> uraw = u.data();

        raw_unknowns<const real *> fxraw  = fx.data();
        raw_unknowns<const real *> fyraw  = fy.data();

        const sloped<real> *braw = b.data();

        cuda_helper::dim3 block(128);
        size_t stride = block.x - 1;
        cuda_helper::dim3 grid((fx.m() + stride - 1) / stride, ceildiv<32>(n));


        GPU_limit_slopes(grid, block, sizeof(raw_unknowns<real>) * block.x)({
                &m, &n, &ld, &lines, &stride,
                &braw,
                &fxraw,
                &fyraw,
                &uraw
            });
    }

    void add_fluxes_and_rhs(const real dt, const real hx, const real hy, const gpu_unknowns &u0, const gpu_sloped_array &b,
            const gpu_flux &fx, const gpu_flux &fy, gpu_unknowns &u)
    {
        size_t m  = u.m() - 2;
        size_t n  = u.n() - 2;
        size_t ld = u.ld();

        size_t lines = 32;

        raw_unknowns<const sloped<real> *> u0raw = u0.data();
        const sloped<real> *braw  = b.data();
        raw_unknowns<sloped<real> *> uraw = u.data();

        raw_unknowns<const real *> fxraw  = fx.data();
        raw_unknowns<const real *> fyraw  = fy.data();

        cuda_helper::dim3 block(128);
        size_t stride = block.x - 1;
        cuda_helper::dim3 grid((fx.m() + stride - 1) / stride, ceildiv<32>(n));

        GPU_add_flux(grid, block, sizeof(raw_unknowns<real>) * block.x)({
                &m, &n, &ld, &lines, &stride,
                &dt, &hx, &hy,
                &u0raw, &braw,
                &fxraw, &fyraw, &uraw
            });
    }
    real compute_max_speed(const gpu_unknowns &u) {
        size_t m  = u.m() - 2;
        size_t n  = u.n() - 2;
        size_t ld = u.ld();

        size_t lines = 32;

        raw_unknowns<const sloped<real> *> uraw = u.data();

        cuda_helper::allocator<float> retalloc;
        float *ret = retalloc.allocate(1);
        CUDA_CHECK(cuMemsetD32(reinterpret_cast<CUdeviceptr>(ret), 0, sizeof(float) / 4));

        cuda_helper::dim3 block(128);
        cuda_helper::dim3 grid(ceildiv<128>(m + 1), ceildiv<32>(n));

        GPU_max_speed(grid, block, sizeof(real) * block.x)({
                &m, &n, &ld, &lines,
                &uraw, &ret
            });

        float max_speed;
        memcpy_DtoH(&max_speed, ret, sizeof(float));

        retalloc.deallocate(ret, 1);

        return max_speed;
    }
};

#endif
