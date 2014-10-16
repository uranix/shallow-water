extern "C" __global__ void sum(const int N, const float *a, const float *b, float *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

extern "C" __global__ void scal(const int N, float *a, float m) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        a[i] *= m;
}
