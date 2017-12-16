#include "common.hpp"
#include<random>
#include <typeinfo>

#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

template< typename T >
int check(T result, const char * const func, const char * const file, const int line)
{
	if (result) {
		fprintf(stderr, "Error: at %s: %d, error code=%d, func: %s\n", file, line, static_cast<unsigned int>(result), func);
		return -1;
	}
}

template< typename T >
int check_Cuda(T result, const char * const func, const char * const file, const int line)
{
	if (result) {
		fprintf(stderr, "Error CUDA: at %s: %d, error code=%d, func: %s\n", file, line, static_cast<unsigned int>(result), func);
		cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
		return -1;
	}
}

void generator_random_number(float* data, int length, float a, float b)
{
	std::random_device rd; std::mt19937 generator(rd()); // 每次产生不固定的不同的值
	//std::default_random_engine generator; // 每次产生固定的不同的值
	std::uniform_real_distribution<float> distribution(a, b);
	for (int i = 0; i < length; ++i) {
		data[i] = distribution(generator);
	}
}

template<typename T> // unsigned char, char, int , short
void generator_random_number(T* data, int length, T a, T b)
{
	std::random_device rd; std::mt19937 generator(rd()); // 每次产生不固定的不同的值
	//std::default_random_engine generator; // 每次产生固定的不同的值
	std::uniform_int_distribution<int> distribution(a, b);
	for (int i = 0; i < length; ++i) {
		data[i] = static_cast<T>(distribution(generator));
	}
}

int save_image(const cv::Mat& mat1, const cv::Mat& mat2, int width, int height, const std::string& name)
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
int compare_result(const T* src1, const T* src2, int length)
{
	CHECK(src1);
	CHECK(src2);

	int count{ 0 };
	for (int i = 0; i < length; ++i) {
		if (fabs(double(src1[i] - src2[i])) > EPS_) {
			if (typeid(float).name() == typeid(T).name() || typeid(double).name() == typeid(T).name())
				fprintf(stderr, "Error: index: %d, val1: %f, val2: %f\n", i, src1[i], src2[i]);
			else
				fprintf(stderr, "Error: index: %d, val1: %d, val2: %d\n", i, src1[i], src2[i]);

			++count;
		}

		if (count > 100) return -1;
	}

	if (count == 0) {
		fprintf(stdout, "they are equal\n");
	}

	return 0;
}

///////////////////////////////////////
template int check<int>(int, const char * const, const char * const, const int);
template int check_Cuda(int, const char * const, const char * const, const int);
template void generator_random_number<unsigned char>(unsigned char*, int, unsigned char, unsigned char);
template void generator_random_number<char>(char*, int, char, char);
template void generator_random_number<int>(int*, int, int, int);
template void generator_random_number<short>(short*, int, short, short);
template int compare_result<float>(const float*, const float*, int);
template int compare_result<unsigned char>(const unsigned char*, const unsigned char*, int);
