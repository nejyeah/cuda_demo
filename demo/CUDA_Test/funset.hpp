#ifndef FBC_CUDA_TEST_FUNSET_HPP_
#define FBC_CUDA_TEST_FUNSET_HPP_

int test_vector_add();
int vector_add_cpu(const float* A, const float* B, float* C, int numElements, float* elapsed_time);
int vector_add_gpu(const float* A, const float* B, float* C, int numElements, float* elapsed_time);

#endif // FBC_CUDA_TEST_FUNSET_HPP_
