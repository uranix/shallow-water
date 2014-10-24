#define _SUFFIXED2(real, name) name ## _ ## real
#define _SUFFIXED1(real, name) _SUFFIXED2(real, name)
#define SUFFIXED(name) _SUFFIXED1(real, name)

extern "C" __global__ void SUFFIXED(blend)(
        const size_t m, const size_t n, const size_t ld, const real w,
        raw_unknowns<sloped<real> *> u,
        raw_unknowns<const sloped<real> *> o
    )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t xy = x + ld * y;
    real wu = 1 - w;
    if (x < m && y < n) {
        u.h [xy] = wu * u.h [xy] + w * o.h [xy];
        u.hu[xy] = wu * u.hu[xy] + w * o.hu[xy];
        u.hv[xy] = wu * u.hv[xy] + w * o.hv[xy];
    }
}

__device__ inline void d2s(sloped<real> &val, const real sx, const real sy) {
    sloped<real> z = val;
    z.vx *= sx;
    z.vy *= sy;
    val = z;
}

extern "C" __global__ void SUFFIXED(der2slope)(
        const size_t m, const size_t n, const size_t ld,
        const real hx, const real hy,
        sloped<real> *b, raw_unknowns<sloped<real> * > u
    )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t xy = x + ld * y;

    const real hx2 = hx / 2;
    const real hy2 = hy / 2;

    if (x < m && y < n) {
        d2s(b   [xy], hx2, hy2);
        d2s(u.h [xy], hx2, hy2);
        d2s(u.hu[xy], hx2, hy2);
        d2s(u.hv[xy], hx2, hy2);
    }
}

extern "C" __global__ void SUFFIXED(flux)(
        const size_t m, const size_t n, const size_t ld, const size_t lines,
        raw_unknowns<const sloped<real> *> u,
        raw_unknowns<real *> fx, raw_unknowns<real *> fy
    )
{
}
