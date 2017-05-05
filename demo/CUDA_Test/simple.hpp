#ifndef FBC_CUDA_TEST_SIMPLE_HPP_
#define FBC_CUDA_TEST_SIMPLE_HPP_

// reference: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\0_Simple
int test_vectorAdd();


int vectorAdd_cpu(const float *A, const float *B, float *C, int numElements);

int vectorAdd_gpu(const float *A, const float *B, float *C, int numElements);

#endif // FBC_CUDA_TEST_SIMPLE_HPP_

