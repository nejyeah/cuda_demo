#include "funset.cuh"
#include <stdio.h>

__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

//__global__：从主机上调用并在设备上运行
__global__ void add_blockIdx(int* a, int* b, int* c)
{
	//计算该索引处的数据
	//变量blockIdx，是一个内置变量，在CUDA运行时中已经预先定义了这个变量
	//此变量中包含的值就是当前执行设备代码的线程块的索引
	int tid = blockIdx.x;//this thread handles the data at its thread id
	if (tid < NUM)
		c[tid] = a[tid] + b[tid];
}

//__device__：表示代码将在GPU而不是主机上运行，
//由于此函数已声明为__device__函数，因此只能从其它__device__函数或者
//从__global__函数中调用它们
__device__ int julia(int x, int y) 
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i=0; i<200; i++) {
		a = a * a + c;

		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel_julia(unsigned char *ptr)
{
	//map from blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	//gridDim为内置变量，对所有的线程块来说，gridDim是一个常数，用来保存线程格每一维的大小
	//此处gridDim的值是(DIM, DIM)
	int offset = x + y * gridDim.x;

	//now calculate the value at that position
	int juliaValue = julia(x, y);

	ptr[offset*4 + 0] = 255 * juliaValue;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
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

__global__ void ripple_kernel(unsigned char *ptr, int ticks)
{
	// map from threadIdx/BlockIdx to pixel position
	//将线程和线程块的索引映射到图像坐标
	//对x和y的值进行线性化从而得到输出缓冲区中的一个偏移
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	//生成一个随时间变化的正弦曲线"波纹"
	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f)); 

	ptr[offset*4 + 0] = grey;
	ptr[offset*4 + 1] = grey;
	ptr[offset*4 + 2] = grey;
	ptr[offset*4 + 3] = 255;
}

__global__ void dot_kernel(float *a, float *b, float *c)
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

__global__ void julia_kernel(unsigned char *ptr)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];

	//now calculate the value at that position
	const float period = 128.0f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *(sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

	//removing this syncthreads shows graphically what happens
	//when it doesn't exist.this is an example of why we need it.
	__syncthreads();

	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}

__global__ void RayTracing_kernel(Sphere *s, unsigned char *ptr)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

	float r=0, g=0, b=0;
	float maxz = -INF;

	for (int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	} 

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

__global__ void RayTracing_kernel(unsigned char *ptr)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

	float r=0, g=0, b=0;
	float maxz = -INF;

	for(int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	} 

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

__global__ void Heat_blend_kernel(float *dst, bool dstOut)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM-1) right--; 

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM-1) bottom -= DIM;

	float t, l, c, r, b;

	if (dstOut) {
		//tex1Dfetch是编译器内置函数，从设备内存取纹理
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);

	} else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}

	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void blend_kernel(float *dst, bool dstOut)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float t, l, c, r, b;
	if (dstOut) {
		t = tex2D(texIn2, x, y-1);
		l = tex2D(texIn2, x-1, y);
		c = tex2D(texIn2, x, y);
		r = tex2D(texIn2, x+1, y);
		b = tex2D(texIn2, x, y+1);
	} else {
		t = tex2D(texOut2, x, y-1);
		l = tex2D(texOut2, x-1, y);
		c = tex2D(texOut2, x, y);
		r = tex2D(texOut2, x+1, y);
		b = tex2D(texOut2, x, y+1);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void Heat_copy_const_kernel(float *iptr)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0)
		iptr[offset] = c;
}

__global__ void copy_const_kernel(float *iptr) 
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConstSrc2, x, y);
	if (c != 0)
		iptr[offset] = c;
}

void generate_frame_opengl(uchar4 *pixels, void*, int ticks)
{
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	ripple_kernel_opengl<<<grids, threads>>>(pixels, ticks);
}

__global__ void ripple_kernel_opengl(uchar4 *ptr, int ticks)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));    
	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

__global__ void Heat_blend_kernel_opengl(float *dst, bool dstOut)
{
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM-1) right--; 

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM-1) bottom -= DIM;

	float t, l, c, r, b;
	if (dstOut) {
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);

	} else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void Heat_copy_const_kernel_opengl(float *iptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0)
		iptr[offset] = c;
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

__global__ void singlestream_kernel(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < NUM) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

__global__ void dot_kernel(int size, float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < size) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	//set the cache values
	cache[cacheIndex] = temp;

	//synchronize threads in this block
	__syncthreads();

	//for reductions(归约), threadsPerBlock must be a power of 2 because of the following code
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}