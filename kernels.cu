#include "sloped.h"
#include "unknowns.h"

#include <cstdlib>

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

#define real float

#include "kernels.impl.cu"

#undef real

#define real double

#include "kernels.impl.cu"

#undef real
