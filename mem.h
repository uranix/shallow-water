#ifndef __MEM_H__
#define __MEM_H__

#include "gpu_allocator.h"

#include <cassert>

#define ASSERT assert

template<class elem, class Allocator> class array2d;

struct mem {
    template<class E>
    using  gpu_array = array2d<E, cuda_helper::allocator<E> >;
    template<class E>
    using host_array = array2d<E, std        ::allocator<E> >;

    template<class E>
    static void copy(gpu_array<E> &dst, const host_array<E> &src) {
        ASSERT(dst.m() == src.m());
        ASSERT(dst.n() == src.n());
        ASSERT(dst.ld() == src.ld());
        CUDA_CHECK(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst.data()), src.data(), src.size() * sizeof(E)));
    }

    template<class E>
    static void copy(host_array<E> &dst, const gpu_array<E> &src) {
        ASSERT(dst.m() == src.m());
        ASSERT(dst.n() == src.n());
        ASSERT(dst.ld() == src.ld());
        CUDA_CHECK(cuMemcpyDtoH(dst.data(), reinterpret_cast<CUdeviceptr>(src.data()), src.size() * sizeof(E)));
    }
};


#endif
