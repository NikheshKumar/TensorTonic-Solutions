#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outN = N - kN + 1;
    while (idx < outN){
        float t = 0.0f;
        for(int j=0; j< kN; j++){
            t += input[idx +j] * kernel[j] ;
        }

        output[idx] = t;
        idx++;
    }
    
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
