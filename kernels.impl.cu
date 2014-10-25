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

__device__ inline void reflect_x(raw_unknowns<sloped<real> > &m) {
    // dh/dx = 0, vx = 0, dvy/dx = 0
    m.h .vx = -m.h .vx;
    m.hu.v  = -m.hu.v;
    m.hv.vx = -m.hv.vx;
}

__device__ inline void reflect_y(raw_unknowns<sloped<real> > &m) {
    // dh/dy = 0, dvx/dx = 0, vy = 0
    m.h .vy = -m.h .vy;
    m.hu.vy = -m.hu.vy;
    m.hv.v  = -m.hv.v;
}

__device__ inline raw_unknowns<sloped<real> > get(const raw_unknowns<const sloped<real> *> &u, ptrdiff_t i) {
    return raw_unknowns<sloped<real> >(u.h[i], u.hu[i], u.hv[i]);
}

__device__ inline real SUFFIXED(g)() {
    return sizeof(real) == 4 ? 9.81f : 9.81;
}

__device__ inline void to_flux_x(raw_unknowns<real> &w) {
    real eps = sizeof(real) == 4 ? 1e-6f : 1e-6;
    if (w.h < eps)
        w.h = eps;

    real h = w.h;
    real u = w.hu / h;
    real v = w.hv / h;

    w.h = h * u;
    w.hu = h * u * u + SUFFIXED(g)() * h / 2;
    w.hv = h * u * v;
}

__device__ inline void to_flux_y(raw_unknowns<real> &w) {
    real eps = sizeof(real) == 4 ? 1e-6f : 1e-6;
    if (w.h < eps)
        w.h = eps;

    real h = w.h;
    real u = w.hu / h;
    real v = w.hv / h;

    w.h = h * v;
    w.hu = h * u * v;
    w.hv = h * v * v + SUFFIXED(g)() * h / 2;
}

__device__ inline void flux_x(
        raw_unknowns<real *> fx, ptrdiff_t i,
        const raw_unknowns<sloped<real> > &_left,
        const raw_unknowns<sloped<real> > &_right
    )
{
    raw_unknowns<real>  left(_left .h.rt(), _left .hu.rt(), _left .hv.rt());
    raw_unknowns<real> right(_right.h.lf(), _right.hu.lf(), _right.hv.lf());

    real cl2 = left.h * SUFFIXED(g)();
    real cr2 = right.h * SUFFIXED(g)();
    real c = sqrt(max(cl2, cr2));

    raw_unknowns<real> fl = left;
    raw_unknowns<real> fr = right;

    to_flux_x(fl);
    to_flux_x(fr);

    fx.h [i] = (fl.h  + fr.h ) / 2 - c / 2 * (right.h  - left.h );
    fx.hu[i] = (fl.hu + fr.hu) / 2 - c / 2 * (right.hu - left.hu);
    fx.hv[i] = (fl.hv + fr.hv) / 2 - c / 2 * (right.hv - left.hv);
};

__device__ inline void flux_y(
        raw_unknowns<real *> fy, ptrdiff_t i,
        const raw_unknowns<sloped<real> > &_bottom,
        const raw_unknowns<sloped<real> > &_top
    )
{
    raw_unknowns<real> bottom(_bottom.h.up(), _bottom.hu.up(), _bottom.hv.up());
    raw_unknowns<real> top   (_top   .h.dn(), _top   .hu.dn(), _top   .hv.dn());

    real cb2 = bottom.h * SUFFIXED(g)();
    real ct2 = top.h * SUFFIXED(g)();
    real c = sqrt(max(cb2, ct2));

    raw_unknowns<real> fb = bottom;
    raw_unknowns<real> ft = top;

    to_flux_y(fb);
    to_flux_y(ft);

    fy.h [i] = (fb.h  + ft.h ) / 2 - c / 2 * (top.h  - bottom.h );
    fy.hu[i] = (fb.hu + ft.hu) / 2 - c / 2 * (top.hu - bottom.hu);
    fy.hv[i] = (fb.hv + ft.hv) / 2 - c / 2 * (top.hv - bottom.hv);
};

extern "C" __global__ void SUFFIXED(flux)(
        const size_t m, const size_t n, const size_t ld, const size_t lines, const size_t stride,
        raw_unknowns<const sloped<real> *> u, raw_unknowns<real *> fx, raw_unknowns<real *> fy
    )
{
    extern __shared__ raw_unknowns<sloped<real> > SUFFIXED(mid)[]; /* IMO bug, extern names are in conflict for different real types */
    raw_unknowns<sloped<real> > *mid = SUFFIXED(mid);
    raw_unknowns<sloped<real> > dn;
    int ylo = 1 + lines * blockIdx.y;
    int yhi = ylo + lines;
    if (yhi > n + 1)
        yhi = n + 1;
    int tx = threadIdx.x;
    int x = threadIdx.x + blockIdx.x * stride;

    if (ylo == 1) {
        // Put u[y = 0] to mid. But, u[y = 0] should be computed from u[y = 1] and b.c.
        if (x < m + 2) {
            mid[tx] = get(u, x + ld);
            reflect_y(mid[tx]);
        }
    }
    for (int y = ylo, yld = ylo * ld; y < yhi; y++, yld += ld) {
        dn = mid[tx];
        if (x >= 1 && x <= m) {
            mid[tx] = get(u, x + yld);
            flux_y(fy, x + yld - ld, dn, mid[tx]);
        }
        __syncthreads();
        if (tx < stride && x <= m) {
            if (x == 0) {
                mid[0] = mid[1];
                reflect_x(mid[0]);
            }
            if (x == m) {
                mid[tx+1] = mid[tx];
                reflect_x(mid[tx+1]);
            }
            flux_x(fx, x + yld - ld, mid[tx], mid[tx+1]);
        }
        __syncthreads();
    }
    if (yhi == n + 1) {
        dn = mid[tx];
        if (x < m + 2) {
            reflect_y(mid[tx]);
            flux_y(fy, x + n * ld, dn, mid[tx]);
        }
    }
}
