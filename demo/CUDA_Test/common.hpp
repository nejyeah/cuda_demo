#ifndef FBC_CUDA_TEST_COMMON_HPP_
#define FBC_CUDA_TEST_COMMON_HPP_

#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#define checkCudaErrors(val) check_Cuda((val), __FUNCTION__, __FILE__, __LINE__)
#define checkErrors(val) check((val), __FUNCTION__, __FILE__, __LINE__)

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

#define PRINT_ERROR_INFO(info) { \
	fprintf(stderr, "Error: %s, file: %s, func: %s, line: %d\n", #info, __FILE__, __FUNCTION__, __LINE__); \
	return -1; }

#define TIME_START_CPU auto start = std::chrono::high_resolution_clock::now();
#define TIME_END_CPU auto end = std::chrono::high_resolution_clock::now(); \
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start); \
	*elapsed_time = duration.count() * 1.0e-6;

#define TIME_START_GPU cudaEvent_t start, stop; /* cudaEvent_t: CUDA event types,结构体类型, CUDA事件,用于测量GPU在某
	个任务上花费的时间,CUDA中的事件本质上是一个GPU时间戳,由于CUDA事件是在
	GPU上实现的,因此它们不适于对同时包含设备代码和主机代码的混合代码计时 */ \
	cudaEventCreate(&start); /* 创建一个事件对象,异步启动 */ \
	cudaEventCreate(&stop); \
	cudaEventRecord(start, 0); /* 记录一个事件,异步启动,start记录起始时间 */
#define TIME_END_GPU cudaEventRecord(stop, 0); /* 记录一个事件,异步启动,stop记录结束时间 */ \
	cudaEventSynchronize(stop); /* 事件同步,等待一个事件完成,异步启动 */ \
	cudaEventElapsedTime(elapsed_time, start, stop); /* 计算两个事件之间经历的时间,单位为毫秒,异步启动 */ \
	cudaEventDestroy(start); /* 销毁事件对象,异步启动 */ \
	cudaEventDestroy(stop);

#define EPS_ 1.0e-4 // ε(Epsilon),非常小的数
#define PI 3.1415926535897932f
#define INF 2.e10f

template< typename T > int check_Cuda(T result, const char * const func, const char * const file, const int line);
template< typename T > int check(T result, const char * const func, const char * const file, const int line);
void generator_random_number(float* data, int length, float a = 0.f, float b = 1.f);
template<typename T> void generator_random_number(T* data, int length, T a = (T)0, T b = (T)1);
int save_image(const cv::Mat& mat1, const cv::Mat& mat2, int width, int height, const std::string& name);
template<typename T> int compare_result(const T* src1, const T* src2, int length);
template<typename T> int read_file(const std::string& name, int length, T* data, int mode = 0); // mode = 0: txt; mode = 1: binary
template<typename T> int write_file(const std::string& name, int length, const T* data, int mode = 0); // mode = 0: txt; mode = 1: binary


#endif // FBC_CUDA_TEST_COMMON_HPP_
