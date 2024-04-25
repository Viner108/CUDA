#include <stdio.h>

// CUDA kernel для умножения матриц
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

// Функиция для запуска матричного умножения на СPU
void matrixMultiplyCUDA(float *h_A, float *h_B, float *h_C, int N){
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

// Выделение памяти на устройстве
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

// Копирование данных на GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// Установка размера блок и сетки
    dim3 threadsPerBlock(16,16);
    dim3 grid((N +threadsPerBlock.x - 1) / threadsPerBlock.x,
              (N +threadsPerBlock.y - 1) / threadsPerBlock.y);
// Запуск ядра
    matMulKernel<<<grid, threadsPerBlock>>>(d_A, d_B, d_C, N);

// Копирование результата обратно в память хоста
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

// Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    // Размер матрицы
    int N = 1024;
    size_t size = N * N * sizeof(float);

    // Выделение памяти для матриц на хосте
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Заполнение матриц и инициализация ( пропущено для краткости)

    //Вызов функции умножения матриц
    matrixMultiplyCUDA(h_A,h_B,h_C,N);

    // Освобождение памяти на хосте
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}