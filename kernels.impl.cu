#define SUFFIXED2(real, name) name ## _ ## real
#define SUFFIXED1(real, name) SUFFIXED2(real, name)
#define SUFFIXED(name) SUFFIXED1(real, name)

extern "C" __global__ void SUFFIXED(blend)(
        const size_t m, const size_t n, const size_t ld, const real w,
        sloped<real> *uh, sloped<real> *uhu, sloped<real> *uhv,
        const sloped<real> *oh, const sloped<real> *ohu, const sloped<real> *ohv
    )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t xy = x + ld * y;
    real wu = 1 - w;
    if (x < m && y < n) {
        uh [xy] = wu * uh [xy] + w * oh [xy];
        uhu[xy] = wu * uhu[xy] + w * ohu[xy];
        uhv[xy] = wu * uhv[xy] + w * ohv[xy];
    }
}

extern "C" __global__ void SUFFIXED(der2slope)(
        const size_t m, const size_t n, const size_t ld,
        const real hx, const real hy,
        sloped<real> *b, sloped<real> *h, sloped<real> *hu, sloped<real> *hv
    )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t xy = x + ld * y;

    const real hx2 = hx / 2;
    const real hy2 = hy / 2;

    if (x < m && y < n) {
        sloped<real> z = b[xy];
        z.vx *= hx2; z.vy *= hy2;
        b[xy] = z;

        z = h[xy];
        z.vx *= hx2; z.vy *= hy2;
        h[xy] = z;

        z = hu[xy];
        z.vx *= hx2; z.vy *= hy2;
        hu[xy] = z;

        z = hv[xy];
        z.vx *= hx2; z.vy *= hy2;
        hv[xy] = z;
    }
}
