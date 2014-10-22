#include "sloped.h"

#include <cstdlib>

#define real float

#include "kernels.impl.cu"

#undef real

#define real double

#include "kernels.impl.cu"

#undef real
