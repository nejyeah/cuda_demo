#ifndef FBC_CUDA_TEST_FUNSET_HPP_
#define FBC_CUDA_TEST_FUNSET_HPP_

int test_ripple();
int ripple_cpu(unsigned char* ptr, int width, int height, int ticks, float* elapsed_time);
int ripple_gpu(unsigned char* ptr, int width, int height, int ticks, float* elapsed_time);

int test_julia();
int julia_cpu(unsigned char* ptr, int width, int height, float scale, float* elapsed_time);
int julia_gpu(unsigned char* ptr, int width, int height, float scale, float* elapsed_time);

int test_matrix_mul();
int matrix_mul_cpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time);
int matrix_mul_gpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time);

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
