#ifndef FBC_CUDA_TEST_COMMON_HPP_
#define FBC_CUDA_TEST_COMMON_HPP_

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

#define checkCudaErrors(val) check((val), __FUNCTION__, __FILE__, __LINE__)
#define checkErrors(val) check((val), __FUNCTION__, __FILE__, __LINE__)

#define PRINT_ERROR_INFO(info) { \
	fprintf(stderr, "Error: %s, file: %s, func: %s, line: %d\n", #info, __FILE__, __FUNCTION__, __LINE__); \
	return -1; }

#define EXP 1.0e-5

#endif // FBC_CUDA_TEST_COMMON_HPP_
