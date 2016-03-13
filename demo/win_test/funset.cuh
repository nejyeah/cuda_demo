#ifndef _FUNSET_CUH_
#define _FUNSET_CUH_

#include <stdio.h>
#include "cpu_anim.h"

#define NUM  33 * 1024 * 1024//1024*1024//33 * 1024//10
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

//__global__关键字将告诉编译器，函数应该编译为在设备而不是主机上运行
__global__ void add(int a, int b, int* c);
__global__ void add_blockIdx(int* a, int* b, int* c);
__global__ void add_threadIdx(int* a, int* b, int* c);
__global__ void add_blockIdx_threadIdx(int* a, int* b, int* c);

struct cuComplex {
	float r, i;

	__device__ cuComplex(float a, float b) : r(a), i(b)  {}

	__device__ float magnitude2(void) 
	{
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a) 
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ int julia(int x, int y); 
__global__ void kernel_julia(unsigned char *ptr);
__global__ void ripple_kernel(unsigned char *ptr, int ticks);
struct Sphere;

struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;
	Sphere *s;
};

void generate_frame(DataBlock *d, int ticks);
void cleanup(DataBlock *d); 

__global__ void dot_kernel(float *a, float *b, float *c);
__global__ void julia_kernel(unsigned char *ptr);

//通过一个数据结构对球面建模
struct Sphere {
	float r,b,g;
	float radius;
	float x,y,z;
	__device__ float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

//声明为常量内存,__constant__将把变量的访问限制为只读
__constant__ Sphere s[SPHERES];

__global__ void RayTracing_kernel(Sphere *s, unsigned char *ptr);
__global__ void RayTracing_kernel(unsigned char *ptr);

//these exist on the GPU side
texture<float> texConstSrc, texIn, texOut;
texture<float, 2> texConstSrc2, texIn2, texOut2;

//this kernel takes in a 2-d array of floats it updates the value-of-interest by a 
//scaled value based on itself and its nearest neighbors
__global__ void Heat_blend_kernel(float *dst, bool dstOut);
__global__ void blend_kernel(float *dst, bool dstOut);
__global__ void Heat_copy_const_kernel(float *iptr);
__global__ void copy_const_kernel(float *iptr);

struct Heat_DataBlock {
	unsigned char   *output_bitmap;
	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;
	CPUAnimBitmap  *bitmap;

	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;
};

//globals needed by the update routine
struct DataBlock_opengl {
	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;

	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;
};

void Heat_anim_gpu(Heat_DataBlock *d, int ticks);
void anim_gpu(Heat_DataBlock *d, int ticks);
//clean up memory allocated on the GPU
void Heat_anim_exit(Heat_DataBlock *d);
void anim_exit(Heat_DataBlock *d); 
void generate_frame_opengl(uchar4 *pixels, void*, int ticks);
__global__ void ripple_kernel_opengl(uchar4 *ptr, int ticks);
__global__ void Heat_blend_kernel_opengl(float *dst, bool dstOut);
__global__ void Heat_copy_const_kernel_opengl(float *iptr);
void anim_gpu_opengl(uchar4* outputBitmap, DataBlock_opengl *d, int ticks);
void anim_exit_opengl(DataBlock_opengl *d);
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo);
__global__ void singlestream_kernel(int *a, int *b, int *c);
__global__ void dot_kernel(int size, float *a, float *b, float *c);

struct DataStruct{
	int     deviceID;
	int     size;
	float   *a;
	float   *b;
	float   returnValue;
};

#endif //_FUNSET_CUH_
