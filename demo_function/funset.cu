#include <stdio.h>

#include "funset.h"

__global__ void add(int a, int b, int* c)
{
  *c = a + b;
}

__global__ void add_blockIdx(int* a, int* b, int* c)  
{  
  int tid = blockIdx.x;//this thread handles the data at its thread id  
  if (tid < NUM)  
    c[tid] = a[tid] + b[tid];  
}  

__global__ void add_threadIdx(int* a, int* b, int* c)  
{  
  //使用线程索引来对数据进行索引而非通过线程块索引(blockIdx.x)  
  int tid = threadIdx.x;  
  if (tid < NUM)  
    c[tid] = a[tid] + b[tid];  
}  

__global__ void add_blockIdx_threadIdx(int* a, int* b, int* c)  
{  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid == 0) {  
    printf("blockDim.x = %d, gridDim.x = %d\n", blockDim.x, gridDim.x);  
  }  
  while (tid < NUM) {  
    c[tid] = a[tid] + b[tid];  
    tid += blockDim.x * gridDim.x;  
  }  
}  

__global__ void dot_kernel(float* a, float* b, float* c)  
{  
  //声明了一个共享内存缓冲区，它将保存每个线程计算的加和值  
  __shared__ float cache[threadsPerBlock];  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  int cacheIndex = threadIdx.x;  

  float temp = 0;  
  while (tid < NUM) {  
    temp += a[tid] * b[tid];  
    tid += blockDim.x * gridDim.x;  
  }  

  //set the cache values  
  cache[cacheIndex] = temp;  

  //synchronize threads in this block  
  //对线程块中的线程进行同步  
  //这个函数将确保线程块中的每个线程都执行完__syncthreads()前面的语句后，才会执行下一条语句  
  __syncthreads();  

  //for reductions(归约), threadsPerBlock must be a power of 2 because of the following code  
  int i = blockDim.x/2;  
  while (i != 0) {  
    if (cacheIndex < i)  
      cache[cacheIndex] += cache[cacheIndex + i];  
    //在循环迭代中更新了共享内存变量cache，并且在循环的下一次迭代开始之前，  
    //需要确保当前迭代中所有线程的更新操作都已经完成  
    __syncthreads();  
    i /= 2;  
  }  
  if (cacheIndex == 0)  
    c[blockIdx.x] = cache[0];  
}  

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)  
{  
    //clear out the accumulation buffer called temp since we are launched with 256 threads,   
    //it is easy to clear that memory with one write per thread  
    __shared__  unsigned int temp[256]; //共享内存缓冲区  
    temp[threadIdx.x] = 0;  
    __syncthreads();  
  
    //calculate the starting index and the offset to the next block that each thread will be processing  
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    int stride = blockDim.x * gridDim.x;  
    while (i < size) {  
        atomicAdd(&temp[buffer[i]], 1);  
        i += stride;  
    }  
  
    //sync the data from the above writes to shared memory then add the shared memory values to the values from  
    //the other thread blocks using global memory atomic adds same as before, since we have 256 threads,  
    //updating the global histogram is just one write per thread!  
    __syncthreads();  
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);  
}  

