extern "C" {
    __global__ void saxpy(uint n, float* const x, float alpha, float* __restrict__ y) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            y[idx] += alpha * x[idx];
        }
    }
}
