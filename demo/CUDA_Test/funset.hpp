#ifndef FBC_CUDA_TEST_FUNSET_HPP_
#define FBC_CUDA_TEST_FUNSET_HPP_

#include <cstdlib>
#include <vector>

int test_layer_reverse();
int layer_reverse_cpu(const float* src, float* dst, int length, const std::vector<int>& vec, float* elapsed_time);
int layer_reverse_gpu(const float* src, float* dst, int length, const std::vector<int>& vec, float* elapsed_time);

int test_layer_channel_normalize();
int layer_channel_normalize_cpu(const float* src, float* dst, int width, int height, int channels, float* elapsed_time);
int layer_channel_normalize_gpu(const float* src, float* dst, int width, int height, int channels, float* elapsed_time);

int test_get_device_info();
int get_device_info();

int test_matrix_mul();
int matrix_mul_cpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time);
int matrix_mul_gpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time);

int test_streams();
int streams_cpu(const int* a, const int* b, int* c, int length, float* elapsed_time);
int streams_gpu(const int* a, const int* b, int* c, int length, float* elapsed_time);

int test_calculate_histogram();
int calculate_histogram_cpu(const unsigned char* data, int length, unsigned int* hist, unsigned int& value, float* elapsed_time);
int calculate_histogram_gpu(const unsigned char* data, int length, unsigned int * hist, unsigned int& value, float* elapsed_time);

int test_heat_conduction();
int heat_conduction_cpu(unsigned char* ptr, int width, int height, const float* src, float speed, float* elapsed_time);
int heat_conduction_gpu(unsigned char* ptr, int width, int height, const float* src, float speed, float* elapsed_time);

int test_ray_tracking();
int ray_tracking_cpu(const float* a, const float* b, const float* c, int sphere_num, unsigned char* ptr, int width, int height, float* elapsed_time);
int ray_tracking_gpu(const float* a, const float* b, const float* c, int sphere_num, unsigned char* ptr, int width, int height, float* elapsed_time);

int test_green_ball();
int green_ball_cpu(unsigned char* ptr, int width, int height, float* elapsed_time);
int green_ball_gpu(unsigned char* ptr, int width, int height, float* elapsed_time);

int test_ripple();
int ripple_cpu(unsigned char* ptr, int width, int height, int ticks, float* elapsed_time);
int ripple_gpu(unsigned char* ptr, int width, int height, int ticks, float* elapsed_time);

int test_julia();
int julia_cpu(unsigned char* ptr, int width, int height, float scale, float* elapsed_time);
int julia_gpu(unsigned char* ptr, int width, int height, float scale, float* elapsed_time);

int test_dot_product();
int dot_product_cpu(const float* A, const float* B, float* value, int elements_num, float* elapsed_time);
int dot_product_gpu(const float* A, const float* B, float* value, int elements_num, float* elapsed_time);

int test_long_vector_add();
int long_vector_add_cpu(const float* A, const float* B, float* C, int elements_num, float* elapsed_time);
int long_vector_add_gpu(const float* A, const float* B, float* C, int elements_num, float* elapsed_time);

int test_vector_add();
int vector_add_cpu(const float* A, const float* B, float* C, int numElements, float* elapsed_time);
int vector_add_gpu(const float* A, const float* B, float* C, int numElements, float* elapsed_time);

#endif // FBC_CUDA_TEST_FUNSET_HPP_
