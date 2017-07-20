#ifndef FBC_CUDA_TEST_COMMON_HPP_
#define FBC_CUDA_TEST_COMMON_HPP_

#include<random>

template< typename T >
static inline int check_Cuda(T result, const char * const func, const char * const file, const int line)
{
	if (result) {
		fprintf(stderr, "Error CUDA: at %s: %d, error code=%d, func: %s\n", file, line, static_cast<unsigned int>(result), func);
		cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
		return -1;
	}
}

template< typename T >
static inline int check(T result, const char * const func, const char * const file, const int line)
{
	if (result) {
		fprintf(stderr, "Error: at %s: %d, error code=%d, func: %s\n", file, line, static_cast<unsigned int>(result), func);
		return -1;
	}
}

#define checkCudaErrors(val) check_Cuda((val), __FUNCTION__, __FILE__, __LINE__)
#define checkErrors(val) check((val), __FUNCTION__, __FILE__, __LINE__)

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

#define PRINT_ERROR_INFO(info) { \
	fprintf(stderr, "Error: %s, file: %s, func: %s, line: %d\n", #info, __FILE__, __FUNCTION__, __LINE__); \
	return -1; }

#define EPS 1.0e-4 // ε(Epsilon),非常小的数

static inline void generator_random_number(float* data, int length, float a = 0.f, float b = 1.f)
{
	std::random_device rd; std::mt19937 generator(rd()); // 每次产生不固定的不同的值
	//std::default_random_engine generator; // 每次产生固定的不同的值
	std::uniform_real_distribution<float> distribution(a, b);
	for (int i = 0; i < length; ++i) {
		data[i] = distribution(generator);
	}
}

#endif // FBC_CUDA_TEST_COMMON_HPP_
