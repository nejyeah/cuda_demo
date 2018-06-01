#include <stdio.h>
#include "task1.h"
#include "common_c.h"
#include "common_cuda.h"

__global__ void foo (void) {
  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;
  __syncthreads();
  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];
  // device function, common_cuda.cu
  bar();
}

int task1 (void) {
  // common_c.h and common_c.cpp
  print_c();

  unsigned int i;
  int *dg, hg[N];
  int sum = 0;
  foo<<<1, N>>>();
  if(cudaGetSymbolAddress((void**)&dg, g)){
    printf("couldn't get the symbol addr\n");
    return 1;
  }
  if(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost)){
    printf("couldn't memcpy\n");
    return 1;
  }
  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }
}
