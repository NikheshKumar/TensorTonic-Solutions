#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    // Write code here
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int H_out = H - kH + 1;
    int W_out = W - kW + 1;
    if(i<H_out && j<W_out){
        float running_sum = 0.0f;
        for(int a = 0; a < kH; a++){
            for(int b=0; b< kW; b++){
                running_sum += input[(i+a)*W + (j+b)] * kernel[a*kW + b]; 
            }
        }
        output[i*W_out + j] = running_sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}
