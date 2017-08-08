#include "funset.hpp"
#include <iostream>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

__global__ static void image_normalize(const float* src, float* dst, int count, int offset)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index > count - 1) return;

	const float* input = src + index * offset;
	float* output = dst + index * offset;
	float mean{ 0.f }, sd{ 0.f };

	for (size_t i = 0; i < offset; ++i) {
		mean += input[i];
		sd += pow(input[i], 2.f);
		output[i] = input[i];
	}

	mean /= offset;
	sd /= offset;
	sd -= pow(mean, 2.f);
	sd = sqrt(sd);
	if (sd < EPS_) sd = 1.f;

	for (size_t i = 0; i < offset; ++i) {
		output[i] = (input[i] - mean) / sd;
	}
}

int image_normalize_gpu(const float* src, float* dst, int width, int height, int channels, float* elapsed_time)
{
	/* cudaEvent_t: CUDA event types,结构体类型, CUDA事件,用于测量GPU在某
	个任务上花费的时间,CUDA中的事件本质上是一个GPU时间戳,由于CUDA事件是在
	GPU上实现的,因此它们不适于对同时包含设备代码和主机代码的混合代码计时 */
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象,异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件,异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	float *dev_src{ nullptr }, *dev_dst{ nullptr };
	size_t length{ width * height * channels * sizeof(float) };

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&dev_src, length);
	cudaMalloc(&dev_dst, length);

	/* cudaMemcpy: 在主机端和设备端拷贝数据,此函数第四个参数仅能是下面之一:
	(1). cudaMemcpyHostToHost: 拷贝数据从主机端到主机端
	(2). cudaMemcpyHostToDevice: 拷贝数据从主机端到设备端
	(3). cudaMemcpyDeviceToHost: 拷贝数据从设备端到主机端
	(4). cudaMemcpyDeviceToDevice: 拷贝数据从设备端到设备端
	(5). cudaMemcpyDefault: 从指针值自动推断拷贝数据方向,需要支持
	统一虚拟寻址(CUDA6.0及以上版本)
	cudaMemcpy函数对于主机是同步的 */
	cudaMemcpy(dev_src, src, length, cudaMemcpyHostToDevice);

	image_normalize << < 2, 256 >> >(dev_src, dev_dst, channels, width*height);

	cudaMemcpy(dst, dev_dst, length, cudaMemcpyDeviceToHost);

	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(dev_src);
	cudaFree(dev_dst);

	// cudaEventRecord: 记录一个事件,异步启动,stop记录结束时间
	cudaEventRecord(stop, 0);
	// cudaEventSynchronize: 事件同步,等待一个事件完成,异步启动
	cudaEventSynchronize(stop);
	// cudaEventElapseTime: 计算两个事件之间经历的时间,单位为毫秒,异步启动
	cudaEventElapsedTime(elapsed_time, start, stop);
	// cudaEventDestroy: 销毁事件对象,异步启动
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

