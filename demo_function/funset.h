#ifndef _FUNSET_H_
#define _FUNSET_H_

#include <stdio.h>

//#include "cpu_bitmap.h"

static void HandleError(cudaError_t err, const char *file, int line){
  if(err != cudaSuccess){
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))

//#define NUM 33 * 1024 * 1024 // i*i时溢出
#define NUM 33 * 1024
#define DIM 1024//1000  
#define PI 3.1415926535897932f  
#define imin(a, b) (a < b ? a : b)  
const int threadsPerBlock = 256;  
const int blocksPerGrid = imin(32, (NUM/2+threadsPerBlock-1) / threadsPerBlock);//imin(32, (NUM + threadsPerBlock - 1) / threadsPerBlock);  
#define rnd(x) (x * rand() / RAND_MAX)  
#define INF 2e10f  
#define SPHERES 20  
#define MAX_TEMP 1.0f  
#define MIN_TEMP 0.0001f  
#define SPEED 0.25f  
#define SIZE (100*1024*1024)  
#define FULL_DATA_SIZE (NUM*20)  

__global__ void add(int a, int b, int* c);
__global__ void add_blockIdx(int* a, int* b, int* c);  
__global__ void add_threadIdx(int* a, int* b, int* c);  
__global__ void add_blockIdx_threadIdx(int* a, int* b, int* c);  
__global__ void dot_kernel(float* a, float* b, float* c);  
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo);  
#endif
