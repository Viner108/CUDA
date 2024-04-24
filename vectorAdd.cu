#include <stdio.h>              // Включаем стандартную библиотеку ввода-вывода
#include <cuda_runtime.h>       // Включаем заголовочный файл для работы с CUDA

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // Вычисляем индекс текущего элемента в массиве
    if (i < numElements) {                        // Проверяем, не вышли ли за границы массива
        C[i] = A[i] + B[i];                       // Выполняем операцию сложения для текущего элемента
    }
}

int main() {
    int numElements = 50000;                     // Количество элементов в векторах
    size_t size = numElements * sizeof(float);   // Вычисляем размер в байтах для массивов векторов
    float *h_A = (float *)malloc(size);          // Выделяем память под вектор A на хосте
    float *h_B = (float *)malloc(size);          // Выделяем память под вектор B на хосте
    float *h_C = (float *)malloc(size);          // Выделяем память под вектор C на хосте

    // Инициализируем векторы A и B случайными значениями
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;       // Заполняем вектор A случайными значениями
        h_B[i] = rand() / (float)RAND_MAX;       // Заполняем вектор B случайными значениями
    }

    float *d_A = NULL;                            // Указатель на устройство для вектора A
    cudaMalloc((void**)&d_A, size);               // Выделяем память на устройстве для вектора A
    float *d_B = NULL;                            // Указатель на устройство для вектора B
    cudaMalloc((void**)&d_B, size);               // Выделяем память на устройстве для вектора B
    float *d_C = NULL;                            // Указатель на устройство для вектора C
    cudaMalloc((void**)&d_C, size);               // Выделяем память на устройстве для вектора C

    // Копируем векторы A и B с хоста на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  // Копируем вектор A с хоста на устройство
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  // Копируем вектор B с хоста на устройство

    int threadsPerBlock = 256;                          // Количество потоков в блоке
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  // Количество блоков в сетке

    // Запускаем ядро vectorAdd для выполнения сложения векторов на устройстве
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Копируем вектор C с устройства на хост
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);   // Копируем вектор C с устройства на хост

    // Проверяем результат сложения на соответствие ожидаемому
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {      // Проверяем ошибку суммы с ожидаемым результатом
            fprintf(stderr,"Результат не соответсвует ожидаемому!\n");  // Выводим сообщение об ошибке
            exit(EXIT_FAILURE);                            // Завершаем программу с ошибкой
        }
    }
    
    printf("Сложение векторов выполнено успешно.\n");     // Выводим сообщение об успешном завершении

    // Освобождаем память на устройстве
    cudaFree(d_A);             // Освобождаем память для вектора A
    cudaFree(d_B);             // Освобождаем память для вектора B
    cudaFree(d_C);             // Освобождаем память для вектора C

    free(h_A);                 // Освобождаем память для вектора A на хосте
    free(h_B);                 // Освобождаем память для вектора B на хосте
    free(h_C);                 // Освобождаем память для вектора C на хосте

    cudaDeviceReset();         // Сбрасываем состояние устройства CUDA

    return 0;                  // Возвращаем успешный код завершения программы
}