#ifndef FBC_CUDA_TEST_COMMON_HPP_
#define FBC_CUDA_TEST_COMMON_HPP_

#include<random>
#include <opencv2/opencv.hpp>

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

#define EPS_ 1.0e-4 // ε(Epsilon),非常小的数
#define PI 3.1415926535897932f
#define INF 2.e10f

static inline void generator_random_number(float* data, int length, float a = 0.f, float b = 1.f)
{
	std::random_device rd; std::mt19937 generator(rd()); // 每次产生不固定的不同的值
	//std::default_random_engine generator; // 每次产生固定的不同的值
	std::uniform_real_distribution<float> distribution(a, b);
	for (int i = 0; i < length; ++i) {
		data[i] = distribution(generator);
	}
}

template<typename T> // unsigned char, char, int , short
static inline void generator_random_number(T* data, int length, T a = (T)0, T b = (T)1)
{
	std::random_device rd; std::mt19937 generator(rd()); // 每次产生不固定的不同的值
	//std::default_random_engine generator; // 每次产生固定的不同的值
	std::uniform_int_distribution<int> distribution(a, b);
	for (int i = 0; i < length; ++i) {
		data[i] = static_cast<T>(distribution(generator));
	}
}

static int save_image(const cv::Mat& mat1, const cv::Mat& mat2, int width, int height, const std::string& name)
{
	CHECK(mat1.type() == mat2.type());

	cv::Mat src1, src2, dst;

	cv::resize(mat1, src1, cv::Size(width / 2, height));
	cv::resize(mat2, src2, cv::Size(width / 2, height));
	dst = cv::Mat(height, width / 2 * 2, mat1.type());
	cv::Mat tmp = dst(cv::Rect(0, 0, width / 2, height));
	src1.copyTo(tmp);
	tmp = dst(cv::Rect(width / 2, 0, width / 2, height));
	src2.copyTo(tmp);

	cv::imwrite(name, dst);
}

template<typename T>
static inline int compare_result(const T* src1, const T* src2, int length)
{
	CHECK(src1);
	CHECK(src2);

	int count{ 0 };
	for (int i = 0; i < length; ++i) {
		if (fabs(src1[i] - src2[i]) > EPS_) {
			if (typeid(float).name() == typeid(T).name() || typeid(double).name() == typeid(T).name())
				fprintf(stderr, "index: %d, val1: %f, val2: %f\n", i, src1[i], src2[i]);
			else
				fprintf(stderr, "index: %d, val1: %d, val2: %d\n", i, src1[i], src2[i]);

			++count;
		}

		if (count > 100) return -1;
	}

	return 0;
}

#endif // FBC_CUDA_TEST_COMMON_HPP_
