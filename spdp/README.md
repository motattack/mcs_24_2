# Суперкомпьютеры и параллельная обработка данных [🔙](https://github.com/motattack/mcs_24_2)

## ⚡ CUDA [Кластер](https://cc.dvfu.ru/)
Базовые команды для работы на кластере.

### 📦 Загрузка модуля CUDA:
```shell
module load cuda
```

### 🛠️ Компилятор:
```shell
mpic++
```

### 🚀 Запуск на нескольких потоках:
```shell
mpiexec -np 
```

### 📖 Книги:
1. [CUDA C++ Programming Guide](lessons/books/CUDA_guide.pdf)
2. [Параллельные вычисления на GPU - Боресков](lessons/books/Boreskov_Parallelnye-vychisleniya-na-GPU-Arhitektura-i-programmnaya-model-CUDA.pdf)
3. [CUDA. API - Романенко](lessons/books/4ICaG_2.pdf)
4. [Основы работы с технологией CUDA - Боресков](lessons/books/Boreskov_A_V_,_Kharlamov_A_A._Introduction_to_CUDA_Technology_2010.pdf)
5. [Программная архитектура CUDA - Калгин](lessons/books/cuda-2-program-arch.pdf)
6. [Технология CUDA в примерах - Сандерс, 2013](lessons/books/Sanders_J_,_Kandrot_E._CUDA_by_Example_2013.pdf)
7. [🇬🇧 CUDA Programming - Cook](lessons/books/Shane_Cook_—_CUDA_Programming_A_Developer's_Guide_to_Parallel_Computing.pdf)
8. [🇬🇧 Deep Belief Nets in C++ and CUDA C: Volume 3 - Timothy](lessons/books/Masters_T_Deep_Belief_Nets_in_C++_and_CUDA_C_Volume_3_Convolutional.pdf)
9. [🇬🇧 Parallel Computing for Data Science - Norman](lessons/books/Norman_Matloff_Parallel_Computing_for_Data_Science_With_Examples.pdf)
10. [🇬🇧 CUDA Application Design and Development - Farber](lessons/books/Rob_Farber_—_CUDA_Application_Design_and_Development_—_2011.pdf)
11. [🇬🇧 PROFESSIONAL CUDA C Programming - Cheng](lessons/books/Cheng_J_,_Grossman_M_,_McKercher_T_Professional_CUDA_C_Programming.pdf)
---


## 📝 Задание 1
Необходимо скомпилировать и запустить следующий код:

```cu
// файл: "hw.cu"
#include <iostream>

int main(int argc, char ** argv) {
  std::cout << "Hello, world!\n";

  return 0;
}
```

А так же:

```cu
// файл: "hw_d.cu"
#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>

using namespace std;

__global__ void kernel(void) {
  int ID = blockIdx.x * blockDim.x + threadIdx.x;

  if (ID == 9) {
    printf("blockDim.x %d\n", blockDim.x);
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    printf("identificator, or number of thread is %d\n", ID);
  }

  /*	if(ID==10)
  	{	
        printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        printf("Hello from theread %d\n", ID);
      }
  */
}

int main(void) {
  cout << "Hello, CUDA! \n"; //cpu

  kernel << < 2, 10 >>> (); //gpu 

  cudaDeviceSynchronize();

  return 0;
}
```

---

## 📝 Задание 2
Необходимо реализовать сложение двух массивов на GPU:

```cu
// Kernel definition
global void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

📌 **Замечание:**
```cu
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

См. книгу №1. для подробностей.

---

### 📘 Пример решения:

```cu
// файл: "vecadd_d.cu"
#include <iostream>

#include<stdio.h>

#include <cuda_runtime.h>

#define N 10

__global__ void VecAdd(float * A, float * B, float * C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  float * A = new float[N];
  float * B = new float[N];
  float * C = new float[N];

  for (int i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i * 2.0;
  }

  float * dev_A, * dev_B, * dev_C;
  cudaMalloc((void ** ) & dev_A, sizeof(float) * N);
  cudaMalloc((void ** ) & dev_B, sizeof(float) * N);
  cudaMalloc((void ** ) & dev_C, sizeof(float) * N);

  cudaMemcpy(dev_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Kernel invocation with N threads
  VecAdd << < 1, N >>> (dev_A, dev_B, dev_C);

  cudaDeviceSynchronize();

  cudaMemcpy(C, dev_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    std::cout << C[i] << "\n";
  }

  delete[] A;
  delete[] B;
  delete[] C;

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}
```

---

## 📝 Задание 3
Необходимо реализовать сложение двух 2-мерных массивов на GPU.

```cu
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}

```

См. книгу №1. для подробностей.

---

## 📝 Задание 4
Необходимо модернизовать код из задания 3,
чтобы он работал в стиле гетерогенного программирования.

См. книгу №1. для подробностей.

---

## 📝 Задание 5
Код с использованием генератора случайных чисел.
Заполняем случайными положительными int числами
от 0 до 10 000 000 одномерный массив на GPU.
Вторая функция должна найти минимум.