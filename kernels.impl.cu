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

__device__ inline real SUFFIXED(g)() {
    return static_cast<real>(9.81);
}

__device__ inline void to_flux_x(raw_unknowns<real> &w) {
    real eps = static_cast<real>(1e-6);
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
    real eps = static_cast<real>(1e-6);
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
        ref_unknowns<real *> fx,
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

    *fx.h  = (fl.h  + fr.h ) / 2 - c / 2 * (right.h  - left.h );
    *fx.hu = (fl.hu + fr.hu) / 2 - c / 2 * (right.hu - left.hu);
    *fx.hv = (fl.hv + fr.hv) / 2 - c / 2 * (right.hv - left.hv);
};

__device__ inline void flux_y(
        ref_unknowns<real *> fy,
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

    *fy.h  = (fb.h  + ft.h ) / 2 - c / 2 * (top.h  - bottom.h );
    *fy.hu = (fb.hu + ft.hu) / 2 - c / 2 * (top.hu - bottom.hu);
    *fy.hv = (fb.hv + ft.hv) / 2 - c / 2 * (top.hv - bottom.hv);
};

extern "C" __global__ void SUFFIXED(flux)(
        const size_t m, const size_t n, const size_t ld, const size_t lines, const size_t stride,
        raw_unknowns<const sloped<real> *> u, raw_unknowns<real *> fx, raw_unknowns<real *> fy
    )
{
    extern __shared__ raw_unknowns<sloped<real> > SUFFIXED(mid)[]; /* IMO bug, extern names are in conflict for different real types */
    raw_unknowns<sloped<real> > *mid = SUFFIXED(mid);
    raw_unknowns<sloped<real> > dn;
    int y_beg = 1 + lines * blockIdx.y;
    int y_end = y_beg + lines;
    if (y_end > n + 1)
        y_end = n + 1;
    int tx = threadIdx.x;
    int x = threadIdx.x + blockIdx.x * stride;

    if (y_beg == 1) {
        // Put u[y = 0] to mid. But, u[y = 0] should be computed from u[y = 1] and b.c.
        if (x < m + 2) {
            mid[tx] = u[x + ld];
            reflect_y(mid[tx]);
        }
    } else
        mid[tx] = u[x + (y_beg - 1) * ld];
    for (int y = y_beg, yld = y_beg * ld; y < y_end; y++, yld += ld) {
        dn = mid[tx];
        if (x >= 1 && x <= m) {
            mid[tx] = u[x + yld];
            flux_y(fy[x + yld - ld], dn, mid[tx]);
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
            flux_x(fx[x + yld - ld], mid[tx], mid[tx+1]);
        }
        __syncthreads();
    }
    if (y_end == n + 1) {
        dn = mid[tx];
        if (x < m + 2) {
            reflect_y(mid[tx]);
            flux_y(fy[x + n * ld], dn, mid[tx]);
        }
    }
}

__device__ inline void integrate(
        const real dt, const real hx, const real hy,
        const ref_unknowns<const sloped<real> *> &u0,
        const raw_unknowns<real> &lf,
        const raw_unknowns<real> &rt,
        const raw_unknowns<real> &dn,
        const raw_unknowns<real> &up,
        const ref_unknowns<sloped<real> *> &u
    )
{
    *u.h  = *u0.h;
    *u.hu = *u0.hu;
    *u.hv = *u0.hv;

    raw_unknowns<real> fx(u.h->v, u.hu->v, u.hv->v);
    raw_unknowns<real> fy(u.h->v, u.hu->v, u.hv->v);
    to_flux_x(fx);
    to_flux_y(fy);

    u.h ->v -= dt * ( (rt.h  - lf.h ) / hx + (up.h  - dn.h ) / hy );
    u.hu->v -= dt * ( (rt.hu - lf.hu) / hx + (up.hu - dn.hu) / hy );
    u.hv->v -= dt * ( (rt.hv - lf.hv) / hx + (up.hv - dn.hv) / hy );

    u.h ->vx -= (dt / 3) * ( (rt.h  + lf.h  - 2 * fx.h ) / hx );
    u.hu->vx -= (dt / 3) * ( (rt.hu + lf.hu - 2 * fx.hu) / hx );
    u.hv->vx -= (dt / 3) * ( (rt.hv + lf.hv - 2 * fx.hv) / hx );

    u.h ->vy -= (dt / 3) * ( (up.h  + dn.h  - 2 * fy.h ) / hy );
    u.hu->vy -= (dt / 3) * ( (up.hu + dn.hu - 2 * fy.hu) / hy );
    u.hv->vy -= (dt / 3) * ( (up.hv + dn.hv - 2 * fy.hv) / hy );
}

extern "C" __global__ void SUFFIXED(add_flux)(
        const size_t m, const size_t n, const size_t ld, const size_t lines, const size_t stride,
        const real dt, const real hx, const real hy,
        raw_unknowns<const sloped<real> *> u0, sloped<real> *b,
        raw_unknowns<const real *> fx, raw_unknowns<const real *> fy, raw_unknowns<sloped<real> *> u
    )
{
    extern __shared__ raw_unknowns<real> SUFFIXED(mid2)[];
    raw_unknowns<real> *mid = SUFFIXED(mid2);
    raw_unknowns<real> dn;
    raw_unknowns<real> up;
    int y_beg = 1 + lines * blockIdx.y;
    int y_end = y_beg + lines;
    if (y_end > n + 1)
        y_end = n + 1;
    int tx = threadIdx.x;
    int x = threadIdx.x + blockIdx.x * stride;

    if (x < m + 2)
        up = fy[x + (y_beg - 1) * ld];

    for (int y = y_beg, yld = y_beg * ld; y < y_end; y++, yld += ld) {
        if (x < m + 2) {
            dn = up;
            up = fy[x + yld];
        }
        if (x <= m)
            mid[tx] = fx[x + yld - ld];
        __syncthreads();
        if (tx > 0 && tx <= stride && x <= m)
            integrate(dt, hx, hy, u0[x + yld], mid[tx-1], mid[tx], dn, up, u[x + yld]);
        __syncthreads();
    }
}

extern "C" __global__ void SUFFIXED(max_speed)(
        const size_t m, const size_t n, const size_t ld, const size_t lines,
        raw_unknowns<const sloped<real> *> u, float *ret
    )
{
    extern __shared__ real SUFFIXED(reduce)[];
    real *reduce = SUFFIXED(reduce);

    int tx = threadIdx.x;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y_beg = 1 + lines * blockIdx.y;
    int y_end = y_beg + lines;
    if (y_end > n + 1)
        y_end = n + 1;

    int yld = y_beg * ld;
    if (x >= 1 && x <= m)
        reduce[tx] = u.h[x + yld].v;
    else
        reduce[tx] = 0;
    yld += ld;

    for (int y = y_beg + 1; y < y_end; y++, yld += ld) {
        real v = 0;
        if (x >= 1 && x <= m)
            v = u.h[x + yld].v;
        if (v > reduce[tx])
            reduce[tx] = v;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tx < s)
            if (reduce[tx + s] > reduce[tx])
                reduce[tx] = reduce[tx + s];
    }

    if (tx == 0) {
        real hmax = reduce[0];
        float cmax = sqrt(hmax * SUFFIXED(g)());
        atomicMax(ret, cmax);
    }
}
