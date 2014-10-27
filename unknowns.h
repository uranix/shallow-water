#ifndef __UNKNOWNS_H__
#define __UNKNOWNS_H__

#ifdef __NVCC__
#define APICALL __device__
#else
#define APICALL
#endif

#include <cstddef>

template<class ComponentType>
struct ref_unknowns;

template<class ComponentType>
struct raw_unknowns {
    ComponentType h, hu, hv;
    APICALL raw_unknowns() { }
    APICALL raw_unknowns(size_t m, size_t n, size_t ld) : h(m, n, ld), hu(m, n, ld), hv(m, n, ld) { }
    APICALL raw_unknowns(const ComponentType &h, const ComponentType &hu, const ComponentType &hv)
        : h(h), hu(hu), hv(hv)
    { }
    APICALL ref_unknowns<ComponentType> operator[](ptrdiff_t i) {
        return ref_unknowns<ComponentType>(h + i, hu + i, hv + i);
    }
    template<typename Y>
    APICALL raw_unknowns &operator=(const ref_unknowns<Y> &other) {
        this->h  = *other.h;
        this->hu = *other.hu;
        this->hv = *other.hv;
        return *this;
    }
};

template<class ComponentType>
struct ref_unknowns {
    ComponentType h, hu, hv;
    APICALL ref_unknowns() { }
    APICALL ref_unknowns(const ComponentType &h, const ComponentType &hu, const ComponentType &hv)
        : h(h), hu(hu), hv(hv)
    { }
    APICALL ref_unknowns &operator=(ref_unknowns &other) {
        *this->h  = *other.h;
        *this->hu = *other.hu;
        *this->hv = *other.hv;
        return *this;
    }
    template<typename Y>
    APICALL ref_unknowns &operator=(raw_unknowns<Y> &other) {
        *this->h  = other.h;
        *this->hu = other.hu;
        *this->hv = other.hv;
        return *this;
    }
};

#ifndef __NVCC__

template<class ComponentType>
struct unknowns : public raw_unknowns<ComponentType> {
    unknowns() { }
    unknowns(size_t m, size_t n, size_t ld) : raw_unknowns<ComponentType>(m, n, ld) { }
    unknowns(ComponentType &h, ComponentType &hu, ComponentType &hv)
        : raw_unknowns<ComponentType>(h, hu, hv)
    { }
    template<class OtherComponentType>
    unknowns &operator=(const unknowns<OtherComponentType> &o) {
        this->h  = o.h;
        this->hu = o.hu;
        this->hv = o.hv;
        return *this;
    }
    raw_unknowns<typename ComponentType::elem_type *> data() {
        typename ComponentType::elem_type *hraw  = this->h .data();
        typename ComponentType::elem_type *huraw = this->hu.data();
        typename ComponentType::elem_type *hvraw = this->hv.data();
        return raw_unknowns<typename ComponentType::elem_type *>(hraw, huraw, hvraw);
    }
    raw_unknowns<const typename ComponentType::elem_type *> data() const {
        const typename ComponentType::elem_type *hraw  = this->h .data();
        const typename ComponentType::elem_type *huraw = this->hu.data();
        const typename ComponentType::elem_type *hvraw = this->hv.data();
        return raw_unknowns<const typename ComponentType::elem_type *>(hraw, huraw, hvraw);
    }
    size_t m() const { return this->h.m(); }
    size_t n() const { return this->h.n(); }
    size_t ld() const { return this->h.ld(); }
};

#endif

#endif
