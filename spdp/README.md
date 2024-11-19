# Суперкомпьютеры и параллельная обработка данных [🔙](https://github.com/motattack/mcs_24_2)

## ⚡ CUDA [Кластер](https://cc.dvfu.ru/)
Базовые команды для работы на кластере.

### 📦 Загрузка модуля CUDA:
```shell
module load cuda
```

### 🛠️ Компиляторы:
```shell
mpic++ или nvcc
```

### 🚀 Запуск:
```shell
mpiexec -np [число потоков]
```

### 📖 Книги:
1. [CUDA C++ Programming Guide](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/CUDA_guide.pdf)
2. [Параллельные вычисления на GPU - Боресков](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Boreskov_Parallelnye-vychisleniya-na-GPU-Arhitektura-i-programmnaya-model-CUDA.pdf)
3. [CUDA. API - Романенко](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/4ICaG_2.pdf)
4. [Основы работы с технологией CUDA - Боресков](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Boreskov_A_V_,_Kharlamov_A_A._Introduction_to_CUDA_Technology_2010.pdf)
5. [Программная архитектура CUDA - Калгин](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/cuda-2-program-arch.pdf)
6. [Технология CUDA в примерах - Сандерс, 2013](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Sanders_J_,_Kandrot_E._CUDA_by_Example_2013.pdf)
7. [🇬🇧 CUDA Programming - Cook](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Shane_Cook_—_CUDA_Programming_A_Developer's_Guide_to_Parallel_Computing.pdf)
8. [🇬🇧 Deep Belief Nets in C++ and CUDA C: Volume 3 - Timothy](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Masters_T_Deep_Belief_Nets_in_C++_and_CUDA_C_Volume_3_Convolutional.pdf)
9. [🇬🇧 Parallel Computing for Data Science - Norman](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Norman_Matloff_Parallel_Computing_for_Data_Science_With_Examples.pdf)
10. [🇬🇧 CUDA Application Design and Development - Farber](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Rob_Farber_—_CUDA_Application_Design_and_Development_—_2011.pdf)
11. [🇬🇧 PROFESSIONAL CUDA C Programming - Cheng](https://raw.githubusercontent.com/motattack/mcs_24_2/main/spdp/lessons/books/Cheng_J_,_Grossman_M_,_McKercher_T_Professional_CUDA_C_Programming.pdf)
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

---

## 📝 Задание 6
Реализовать поиск минимума через разделяемую память (__shared__) в блоке

---

## 📝 Задание 7
Реализовать поиск минимума, используя алгоритм редукции

---

## 📝 Задание 8 - 3.3.2. Пример: транспонирование матрицы
Выполнить транспонирование матрицы

В качестве следующего примера рассмотрим задачу транспонирования квадратной матрицы $A$ размера $N \times N$.
Далее будем считать, что $N$ кратно 16.

Поскольку сама матрица $A$ двумерна, то будет удобно использовать двумерную сетку и двумерные блоки.
В качестве размера блока выберем $16 \times 16$, что позволит запустить до трех блоков на одном мультипроцессоре.
Тогда для транспонирования матрицы можно использовать следующее ядро:
```cuda
__global__ void transpose1(float *inData, float *outData, int n) {
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int inIndex = xIndex + n * yIndex;
    unsigned int outIndex = yIndex + n * xIndex;

    outData[outIndex] = inData[inIndex];
}
```

---

## 📝 Задание 9 - 3.3.3. Пример: перемножение двух матриц
Выполнить перемножение матриц

Несколько более сложным примером (к которому мы также еще вернемся в следующей главе)
будет задача перемножения двух квадратных матриц $A$ и $B$
(также будем считать, что они обе имеют размер $N \times N$,
где $N$ кратно 16). Произведение $C$ двух матриц $A$ и $B$ задается при помощи следующей формулы:

```math
c_{i, j} = \sum_{k=0}^{N-1} a_{i, k} \cdot b_{k, j}.
```

Как и в предыдущем примере, будем использовать двумерные блоки $16 \times 16$ и двумерную сетку. Ниже приводится простейший пример "перемножения в лоб":

```cuda
__global__ void matMult(float *a, float *b, int n, float *c) {
    int bx = blockIdx.x;  // Индекс блока по оси X
    int by = blockIdx.y;  // Индекс блока по оси Y
    int tx = threadIdx.x; // Индекс потока внутри блока по оси X
    int ty = threadIdx.y; // Индекс потока внутри блока по оси Y

    float sum = 0.0f; // Переменная для накопления суммы

    // Вычисление индекса элемента матрицы A
    int ia = n * BLOCK_SIZE * by + n * ty;

    // Вычисление индекса элемента матрицы B
    int ib = BLOCK_SIZE * bx + tx;

    // Перемножение и суммирование
    for (int k = 0; k < n; k++) {
        sum += a[ia + k] * b[ib + k * n];
    }

    // Запись результата в матрицу C
    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic + n * ty + tx] = sum;
}
```

---

## 📝 Задание 10 - 3.4.1. Задача об N-телах
Чтобы проиллюстрировать важность выравнивания, рассмотрим решение на CUDA следующей задачи:  
дано $N$ тел (для простоты считаем, что их массы одинаковы) со своими положениями и скоростями.  
Необходимо просчитать их движение под действием сил взаимного притяжения.

Из школьного курса физики получаем формулу, описывающую полную силу, действующую на $i$-е тело:  
```math
\mathbf{F}_i = \sum_{j=0}^{N} \frac{C}{\lvert \mathbf{p}_j - \mathbf{p}_i \rvert^3} (\mathbf{p}_j - \mathbf{p}_i),
```

где $C$ — константа пропорциональности, зависящая от масс и гравитационной постоянной,  
а $\mathbf{p}_i$ и $\mathbf{p}_j$ — положения $i$-го и $j$-го тел соответственно.

Для реализации моделирования движения тел нам понадобятся четыре массива:  
положения и скорости в текущий и в следующий моменты времени.  
Поскольку мы имеем дело с трехмерными объектами, то проще всего использовать массивы типа `float3`.