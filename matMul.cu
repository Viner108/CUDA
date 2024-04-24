#include <stdio.h>

__global__ void matMulKernel(float *A, float *B, float *C, int N){
    int row =  blockIdx.y * blockDim.y + threadIdx.y;
    int col =  blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(col < N && row < N){
        for (int i = 0; i < N; ++i){
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplyCUDA(float *h_A, float *h_B, float *h_C, int N){
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    
}