#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        output[idx] = (input[idx] * mask[idx])/(1.0f-p);
    }

}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    cudaDeviceSynchronize();
}