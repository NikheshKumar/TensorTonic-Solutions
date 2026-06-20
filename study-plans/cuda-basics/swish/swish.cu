#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx < N){
        float s = expf(input[idx]);
        output[idx] = (s*input[idx]) / (1.0f+s);
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
