#ifndef __SLOPED_H__
#define __SLOPED_H__

#include "array2d.h"

template<typename real>
struct sloped {
    real v;
    real vx;
    real vy;
#if USE_PADDING
    real padding;
#endif
};

template<typename real, class Allocator = std::allocator<sloped<real> > >
using sloped_array = array2d<sloped<real>, Allocator>;

#endif
