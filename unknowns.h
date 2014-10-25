#ifndef __UNKNOWNS_H__
#define __UNKNOWNS_H__

#ifdef __NVCC__
#define APICALL __device__
#else
#define APICALL
#endif

template<class ComponentType>
struct raw_unknowns {
    ComponentType h, hu, hv;
    APICALL raw_unknowns() { }
    APICALL raw_unknowns(size_t m, size_t n) : h(m, n), hu(m, n), hv(m, n) { }
    APICALL raw_unknowns(const ComponentType &h, const ComponentType &hu, const ComponentType &hv)
        : h(h), hu(hu), hv(hv)
    { }
};

#ifndef __NVCC__

template<class ComponentType>
struct unknowns : public raw_unknowns<ComponentType> {
    unknowns() { }
    unknowns(size_t m, size_t n) : raw_unknowns<ComponentType>(m, n) { }
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
