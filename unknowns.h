#ifndef __UNKNOWNS_H__
#define __UNKNOWNS_H__

#include "mem.h"

template<class ComponentType>
struct unknowns {
    ComponentType h, hu, hv;
    unknowns(size_t m, size_t n) : h(m, n), hu(m, n), hv(m, n) { }
    unknowns() { }
    template<class OtherComponentType>
    unknowns &operator=(const unknowns<OtherComponentType> &o) {
        h  = o.h;
        hu = o.hu;
        hv = o.hv;
        return *this;
    }
};

#endif
