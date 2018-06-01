#include <cstdio>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add_gpu(float *dx, float *dy)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  //int temp = dx[id] > 0 ? dx[id]: (-1)*dx[id]; 
  //dy[id] += temp;
  dy[id] += std::abs(dx[id]);
  //dy[id] += dx[id];
}

void add_cpu(int N, float *hx, float *hy)
{
  for(int i=0; i<N; i++){
    hy[i] += std::abs(hx[i]);
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
    hx[i] = 1.0f;
    hy[i] = 2.0f;
  }
  
  cudaMemcpy(dx, hx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, N*sizeof(float), cudaMemcpyHostToDevice);

  // N can be divided by 256
  int threadNums = 256;
  int blockNums = (N + threadNums -1)/threadNums;

  std::chrono::time_point<std::chrono::system_clock> begin; 
  std::chrono::time_point<std::chrono::system_clock> end; 
  std::chrono::duration<double> elapsedTime;
  // call add_cpu
  begin = std::chrono::system_clock::now();
  add_cpu(N, hx, hy); 
  end = std::chrono::system_clock::now();
  elapsedTime = end - begin;
  printf("Call add_cpu, Time: %.6lfs\n", elapsedTime.count());

  // call add_gpu 
  begin = std::chrono::system_clock::now();
  add_gpu<<< blockNums, threadNums>>>(dx, dy);
  end = std::chrono::system_clock::now();
  elapsedTime = end - begin;
  printf("Call add_gpu, Time: %.6lfs\n", elapsedTime.count());

  // block 同步
  cudaDeviceSynchronize();
  
  cudaMemcpy(hy, dy, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i=0; i<N; i++){
    maxError = std::max(maxError, std::abs(hy[i] - 4.0f));
  }
  
  printf("Max error: %.6f\n", maxError);

  delete[] hx;
  delete[] hy;
  cudaFree(dx);
  cudaFree(dy);
  return 0; 
}
