#include <cstdio>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#define sign(x) ((x)>=0?1:-1)

__global__ void sign_gpu_v1(int num, float *dx, float *dy)
{
  CUDA_KERNEL_LOOP(i, num){
    dy[i] = sign(dx[i]);
  }
}

// operate ((x>=0)*2-1) is three times faster than sign function
__global__ void sign_gpu_v2(int num, float *dx, float *dy)
{
  CUDA_KERNEL_LOOP(i, num){
    dy[i] = (dx[i]>=0)*2-1;
  }
}

__global__ void abs_gpu_v1(int num, float *dx, float *dy)
{
  CUDA_KERNEL_LOOP(i, num){
    dy[i] = std::abs(dx[i]);
  }
}

// operate v2 is 2~3 times faster than sign function
__global__ void abs_gpu_v2(int num, float *dx, float *dy)
{
  CUDA_KERNEL_LOOP(i, num){
    dy[i] = ((dx[i]>=0)*2-1)*dx[i];
  }
}

int main()
{
  int N = 1 << 20;
  std::cout<<"N:"<<N<<std::endl;
  float *hx, *hy, *dx, *dy;
  hx = new float[N];
  hy = new float[N];
  cudaMalloc(&dx, N*sizeof(float));
  cudaMalloc(&dy, N*sizeof(float));

  for(int i = 0; i < N; i++){
    hx[i] = (rand()%100)/100.0-0.5;
    hy[i] = (rand()%100)/100.0-0.5;
  }
  
  cudaMemcpy(dx, hx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, N*sizeof(float), cudaMemcpyHostToDevice);

  std::chrono::time_point<std::chrono::system_clock> begin; 
  std::chrono::time_point<std::chrono::system_clock> end; 
  std::chrono::duration<double> elapsedTime;
  // call add_cpu
  begin = std::chrono::system_clock::now();
  //sign_gpu_v1<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, dx, dy);
  abs_gpu_v1<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, dx, dy);
  end = std::chrono::system_clock::now();
  elapsedTime = end - begin;
  printf("Call version 1, Time: %.6lfs\n", elapsedTime.count());

  // call add_gpu 
  begin = std::chrono::system_clock::now();
  //sign_gpu_v2<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, dx, dy);
  abs_gpu_v2<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, dx, dy);
  end = std::chrono::system_clock::now();
  elapsedTime = end - begin;
  printf("Call version 2, Time: %.6lfs\n", elapsedTime.count());

  // block 同步
  cudaDeviceSynchronize();
  
  cudaMemcpy(hy, dy, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<10; i++){
    printf("a: %.2f, b: %.2f", hx[i], hy[i]);
  }

  delete[] hx;
  delete[] hy;
  cudaFree(dx);
  cudaFree(dy);
  return 0; 
}
