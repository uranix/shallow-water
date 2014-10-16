#ifndef __UNKNOWNS_H__
#define __UNKNOWNS_H__

#include "sloped.h"
#include "mem.h"

template<typename real, class Allocator = std::allocator<sloped<real> > >
struct unknowns {
    sloped_array<real, Allocator> h;
    sloped_array<real, Allocator> hu;
    sloped_array<real, Allocator> hv;
    unknowns(size_t m, size_t n) : h(m, n), hu(m, n), hv(m, n) { }

    template<class OtherAllocator>
    unknowns &operator=(const unknowns<real, OtherAllocator> &other) {
        mem::copy(this->h, other.h);
        mem::copy(this->hu, other.hu);
        mem::copy(this->hv, other.hv);

        return *this;
    }

    void blend_with(const real weight, const unknowns &other) {
        throw "not implemented";
    }
};


#endif
