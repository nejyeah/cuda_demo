#include "common.hpp"
#include<random>
#include <fstream>
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

template<typename T> // mode = 0: txt; mode = 1: binary
int read_file(const std::string& name, int length, T* data, int mode)
{
	if (mode == 0) {
		std::ifstream fin(name.c_str(), std::ios::in);
		if (!fin.is_open()) {
			fprintf(stderr, "open file fail: %s\n", name.c_str());
			return -1;
		}

		T* p = data;
		for (int i = 0; i < length; ++i) {
			fin >> p[i];
		}

		fin.close();
	} else {
		std::ifstream fin(name.c_str(), std::ios::in | std::ios::binary);
		if (!fin.is_open()) {
			fprintf(stderr, "open file fail: %s\n", name.c_str());
			return -1;
		}

		T* p = data;
		fin.read((char*)p, sizeof(T)* length);

		fin.close();
	}

	return 0;
}

template<typename T> // mode = 0: txt; mode = 1: binary
int write_file(const std::string& name, int length, const T* data, int mode)
{
	if (mode == 0) {
		std::ofstream fout(name.c_str(), std::ios::out);
		if (!fout.is_open()) {
			fprintf(stderr, "open file fail: %s\n", name.c_str());
			return -1;
		}

		const T* p = data;
		for (int i = 0; i < length; ++i) {
			fout << p[i] << "  ";
		}

		fout.close();
	}
	else {
		std::ofstream fout(name.c_str(), std::ios::out | std::ios::binary);
		if (!fout.is_open()) {
			fprintf(stderr, "open file fail: %s\n", name.c_str());
			return -1;
		}

		const T* p = data;
		fout.write((char*)p, sizeof(T)* length);

		fout.close();
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

template int read_file<float>(const std::string& name, int, float*, int);
template int read_file<int>(const std::string& name, int, int*, int);
template int read_file<unsigned char>(const std::string& name, int, unsigned char*, int);

template int write_file<float>(const std::string& name, int, const float*, int);
template int write_file<int>(const std::string& name, int, const int*, int);
template int write_file<unsigned char>(const std::string& name, int, const unsigned char*, int);

