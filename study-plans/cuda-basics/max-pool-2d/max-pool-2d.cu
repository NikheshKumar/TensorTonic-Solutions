#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    // Write code here
    int i = blockIdx.y * blockDim.y + threadIdx.y ;
    int j = blockIdx.x * blockDim.x + threadIdx.x ; 
    int H_out = (H - kH)/sH + 1;
    int W_out = (W - kW)/sW + 1;
    
    if (i < H_out && j < W_out) {
        float running_max = -FLT_MAX;

        int starting_x = i * sH;
        int starting_y = j * sW;
        
        for (int a = 0; a < kH; a++) {
            for (int b = 0; b < kW; b++) {
                int curr_x = starting_x + a;
                int curr_y = starting_y + b;
                
                if (curr_x >= 0 && curr_x < H && curr_y >= 0 && curr_y < W) {
                    running_max = fmaxf(running_max, input[curr_x * W + curr_y]);
                }
            }
        }
        
    output[i*W_out + j] = running_max;

    }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
    cudaDeviceSynchronize();
}
