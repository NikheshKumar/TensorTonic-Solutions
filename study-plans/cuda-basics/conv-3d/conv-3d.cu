#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    // Write code here

    int outW = W - kW + 1;
    int outH = H - kH + 1;
    int outD = D - kD + 1;

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    float curr = 0.0f;

    if (h >= outH || w >= outW || d >= outD) return;

    for(int kd=0;kd<kD;kd++){
        for(int kh=0;kh<kH;kh++){
            for(int kw=0;kw<kW;kw++){
                float x = input[ ((d + kd) * H + (h + kh)) * W + (w + kw) ];
                float k = kernel[ ((kd * kH) + kh) * kW + kw ];
                curr += x * k ;
            }   
        }

    }
    output[((d*outH)+h)*outW + w] = curr;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(8, 8, 8);
    dim3 blocks((outW + 7) / 8, (outH + 7) / 8, (outD + 7) / 8);
    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
    cudaDeviceSynchronize();
}
