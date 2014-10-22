#ifndef __SLOPED_H__
#define __SLOPED_H__

#ifdef __NVCC__
#define APICALL __device__
#else
#define APICALL
#endif

template<typename real>
struct sloped {
    real v;
    real vx;
    real vy;
#if USE_PADDING
    real padding;
#endif
    APICALL sloped() : sloped(0, 0, 0) { }
    APICALL sloped(const real v, const real vx, const real vy) : v(v), vx(vx), vy(vy) { }
    APICALL const sloped operator+(const sloped &o) const {
        return sloped(v + o.v, vx + o.vx, vy + o.vy);
    }
    APICALL const sloped operator-(const sloped &o) const {
        return sloped(v - o.v, vx - o.vx, vy - o.vy);
    }
    APICALL const sloped operator*(const real w) const {
        return sloped(w * v, w * vx, w * vy);
    }
    APICALL sloped &operator+=(const sloped &o) {
        v  += o.v;
        vx += o.vx;
        vy += o.vy;
        return *this;
    }
    APICALL sloped &operator-=(const sloped &o) {
        v  -= o.v;
        vx -= o.vx;
        vy -= o.vy;
        return *this;
    }
};

template<typename real>
inline APICALL const sloped<real> operator*(const real w, const sloped<real> &o) {
    return o * w;
}

#ifndef __NVCC__

#include "array2d.h"

template<typename real, class Allocator = std::allocator<sloped<real> > >
using sloped_array = array2d<sloped<real>, Allocator>;

#endif

#endif
