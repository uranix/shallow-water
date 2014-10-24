#ifndef __ARRAY2D_H__
#define __ARRAY2D_H__

#include <cstddef>
#include <memory>

#include "mem.h"

template<size_t align>
constexpr size_t alignup(size_t n) {
    static_assert(!(align & (align - 1)), "ailgn size must be a power of two");
    return (n + align - 1) & ~(align - 1);
}

template<class elem, class Allocator = std::allocator<elem> >
class array2d {
    elem *ptr;
    const size_t _m, _n, _ld;
    Allocator alloc;

public:
    typedef elem elem_type;

    array2d(const size_t m, const size_t n, const Allocator &alloc = Allocator())
        : array2d(m, n, alignup<16>(m), alloc) { }

    array2d(const size_t m, const size_t n, const size_t ld, const Allocator &alloc = Allocator())
        : _m(m), _n(n), _ld(ld), alloc(alloc)
    {
        ptr = this->alloc.allocate(size());
    }

    array2d(const array2d &) = delete;
    array2d(const array2d &&) = delete;

    array2d &operator=(const array2d &o) {
        mem::copy(*this, o);
        return *this;
    }

    template<class OtherAllocator>
    array2d &operator=(const array2d<elem, OtherAllocator> &o) {
        mem::copy(*this, o);
        return *this;
    }

    inline size_t n() const { return _n; }
    inline size_t m() const { return _m; }
    inline size_t ld() const { return _ld; }
    inline size_t size() const { return _n * _ld; }

    elem *data() { return ptr; }
    const elem *data() const { return ptr; }

    elem &operator()(ptrdiff_t i, ptrdiff_t j) { return ptr[i + j * ld()]; }
    const elem &operator()(ptrdiff_t i, ptrdiff_t j) const { return ptr[i + j * ld()]; }

    elem &operator[](ptrdiff_t ij) { return ptr[ij]; }
    const elem &operator[](ptrdiff_t ij) const { return ptr[ij]; }

    ~array2d() {
        alloc.deallocate(ptr, size());
    }
};

#endif
