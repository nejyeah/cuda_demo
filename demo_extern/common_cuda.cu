#include"common_cuda.h"
// why define twice, g[N] has aready been defined in h.h
__device__ int g[N];
__device__ void bar (void){
  g[threadIdx.x]++;
}
