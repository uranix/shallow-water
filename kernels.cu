#include "sloped.h"
#include "unknowns.h"

#include <cstdlib>

#define real float

#include "kernels.impl.cu"

#undef real

#define real double

#include "kernels.impl.cu"

#undef real
