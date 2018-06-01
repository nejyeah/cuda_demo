#include "funset.h"
#include <iostream>

using namespace std;

int test1();
int test2();
int test3();//通过线程块索引来计算两个矢量和  
int test4();//Julia的CUDA实现  
int test5();//通过线程索引来计算两个矢量和  
int test6();//通过线程块索引和线程索引来计算两个矢量和  
int test7();//ripple的CUDA实现  
int test8();//点积运算的CUDA实现  
int test9();//Julia的CUDA实现，加入了线程同步函数__syncthreads()  
int test10();//光线跟踪(Ray Tracing)实现，没有常量内存+使用事件来计算GPU运行时间  
int test11();//光线跟踪(Ray Tracing)实现，使用常量内存+使用事件来计算GPU运行时间  
int test12();//模拟热传导，使用纹理内存，有些问题  
int test13();//模拟热传导，使用二维纹理内存，有些问题  
int test14();//ripple的CUDA+OpenGL实现  
int test15();//模拟热传导,CUDA+OpenGL实现，有些问题  
int test16();//直方图计算，利用原子操作函数atomicAdd实现  
int test17();//固定内存的使用  
int test18();//单个stream的使用  
int test19();//多个stream的使用  
int test20();//通过零拷贝内存的方式实现点积运算  
int test21();//使用多个GPU实现点积运算  


int test1()
{
  int a = 2, b = 3, c = 0;
  int* dev_c = NULL;
  cudaError_t cudaStatus;

  cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
  if (cudaStatus != cudaSuccess){
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  add<<<1, 1>>>(a, b, dev_c);
  cudaStatus = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess){
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  printf("%d + %d = %d\n", a, b, c);

  Error:
    cudaFree(dev_c);

  return 0;
}

int test2()  
{  
  int count = -1;  
  HANDLE_ERROR(cudaGetDeviceCount(&count));  
  printf("device count: %d\n", count);  
    
  cudaDeviceProp prop;  
  for (int i = 0; i < count; i++) {  
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));  

    printf("   --- General Information for device %d ---\n", i);  
    printf("Name:  %s\n", prop.name);  
    printf("Compute capability:  %d.%d\n", prop.major, prop.minor);  
    printf("Clock rate:  %d\n", prop.clockRate);  
    printf("Device copy overlap:  ");  
    if (prop.deviceOverlap) printf("Enabled\n");  
    else printf("Disabled\n");  
    printf("Kernel execution timeout :  ");  
    if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");  
    else printf("Disabled\n");  

    printf("   --- Memory Information for device %d ---\n", i);  
    printf("Total global mem:  %ld\n", prop.totalGlobalMem);  
    printf("Total constant Mem:  %ld\n", prop.totalConstMem);  
    printf("Max mem pitch:  %ld\n", prop.memPitch);  
    printf("Texture Alignment:  %ld\n", prop.textureAlignment);  

    printf("   --- MP Information for device %d ---\n", i);  
    printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);  
    printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);  
    printf("Registers per mp:  %d\n", prop.regsPerBlock);  
    printf("Threads in warp:  %d\n", prop.warpSize);  
    printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);  
    printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0],   
        prop.maxThreadsDim[1], prop.maxThreadsDim[2]);  
    printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0],   
        prop.maxGridSize[1], prop.maxGridSize[2]);  
    printf("\n");  
  }  

  int dev;  

  HANDLE_ERROR(cudaGetDevice(&dev));  
  printf("ID of current CUDA device:  %d\n", dev);  

  memset(&prop, 0, sizeof(cudaDeviceProp));  
  prop.major = 1;  
  prop.minor = 3;  
  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));  
  printf("ID of CUDA device closest to revision %d.%d:  %d\n", prop.major, prop.minor, dev);  

  HANDLE_ERROR(cudaSetDevice(dev));  

  return 0;  
}  


int test3()  
{  
  int a[NUM] = {0}, b[NUM] = {0}, c[NUM] = {0};  
  int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;  

  //allocate the memory on the GPU  
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  

  //fill the arrays 'a' and 'b' on the CPU  
  for (int i=0; i<NUM; i++) {  
    a[i] = -i;  
    b[i] = i * i;  
  }  
  cout<<"NUM:"<<NUM<<endl;
  cout<<"b[NUM-1]:"<<b[NUM-1]<<endl;
  cout<<"sizeof(int):"<<sizeof(int)<<endl;

  //copy the arrays 'a' and 'b' to the GPU  
  HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  

  //尖括号中的第一个参数表示设备在执行核函数时使用的并行线程块的数量  
  add_blockIdx<<<NUM,1>>>( dev_a, dev_b, dev_c );  

  //copy the array 'c' back from the GPU to the CPU  
  HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  

  //display the results  
  //for (int i=0; i<NUM; i++) {  
  //    printf( "%d + %d = %d\n", a[i], b[i], c[i] );  
  //}  
  printf( "%d + %d = %d\n", a[NUM-1], b[NUM-1], c[NUM-1] );  

  //free the memory allocated on the GPU  
  HANDLE_ERROR(cudaFree(dev_a));  
  HANDLE_ERROR(cudaFree(dev_b));  
  HANDLE_ERROR(cudaFree(dev_c));  

  return 0;  
}  

int test5()  
{  
  int a[NUM], b[NUM], c[NUM];  
  int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;  

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  

  for (int i = 0; i < NUM; i++) {  
      a[i] = i;  
      b[i] = i * i;  
  }  

  HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  add_threadIdx<<<1, NUM>>>(dev_a, dev_b, dev_c);  
  HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  
  printf("%d + %d = %d\n", a[NUM-1], b[NUM-1], c[NUM-1]);  
  cudaFree(dev_a);  
  cudaFree(dev_b);  
  cudaFree(dev_c);  
  return 0;  
}  

int test6()  
{  
  int a[NUM], b[NUM], c[NUM];  
  int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;  

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  

  for (int i = 0; i < NUM; i++) {  
    a[i] = i;  
    b[i] = i * i / 10;  
  }  

  HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  add_blockIdx_threadIdx<<<128, 128>>>(dev_a, dev_b, dev_c);  
  HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  

  bool success = true;  
  for (int i = 0; i < NUM; i++) {  
    if ((a[i] + b[i]) != c[i]) {  
      printf("error: %d + %d != %d\n", a[i], b[i], c[i]);  
      success = false;  
    }  
  }  

  if (success)  printf("we did it!\n");  
  cudaFree(dev_a);  
  cudaFree(dev_b);  
  cudaFree(dev_c);  
  return 0;  
}  

int test8()  
{  
  float *a, *b, c, *partial_c;  
  float *dev_a, *dev_b, *dev_partial_c;  
  a = (float*)malloc(NUM * sizeof(float));  
  b = (float*)malloc(NUM * sizeof(float));  
  partial_c = (float*)malloc(blocksPerGrid * sizeof(float));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(float)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(float)));  
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));  
  for (int i = 0; i < NUM; i++) {  
    a[i] = i;  
    b[i] = i*2;  
  }  

  HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(float), cudaMemcpyHostToDevice));  
  HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(float), cudaMemcpyHostToDevice));   
  dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);  
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));  

  //finish up on the CPU side  
  c = 0;  
  for (int i = 0; i < blocksPerGrid; i++) {  
    c += partial_c[i];  
  }  
    
  //点积计算结果应该是从0到NUM-1中每个数值的平方再乘以2  
  //闭合形式解  
  #define sum_squares(x)  (x * (x + 1) * (2 * x + 1) / 6)  
  printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(NUM - 1)));  

  HANDLE_ERROR(cudaFree(dev_a));  
  HANDLE_ERROR(cudaFree(dev_b));  
  HANDLE_ERROR(cudaFree(dev_partial_c));  
  free(a);  
  free(b);  
  free(partial_c);  
  return 0;  
}  

// without definition of big_random_block()
//int test16()  
//{  
//  unsigned char *buffer = (unsigned char*)big_random_block(SIZE);  
//  //capture the start time starting the timer here so that we include the cost of  
//  //all of the operations on the GPU.  if the data were already on the GPU and we just   
//  //timed the kernel the timing would drop from 74 ms to 15 ms.  Very fast.  
//  cudaEvent_t start, stop;  
//  HANDLE_ERROR( cudaEventCreate( &start ) );  
//  HANDLE_ERROR( cudaEventCreate( &stop ) );  
//  HANDLE_ERROR( cudaEventRecord( start, 0 ) );  
//
//  // allocate memory on the GPU for the file's data  
//  unsigned char *dev_buffer;  
//  unsigned int *dev_histo;  
//  HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));  
//  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));  
//
//  HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(int)));  
//  HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));  
//
//  //kernel launch - 2x the number of mps gave best timing  
//  cudaDeviceProp prop;  
//  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));  
//  int blocks = prop.multiProcessorCount;  
//  histo_kernel<<<blocks*2, 256>>>(dev_buffer, SIZE, dev_histo);  
//
//  unsigned int histo[256];  
//  HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));  
//
//  //get stop time, and display the timing results  
//  HANDLE_ERROR(cudaEventRecord(stop, 0));  
//  HANDLE_ERROR(cudaEventSynchronize(stop));  
//  float elapsedTime;  
//  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
//  printf("Time to generate:  %3.1f ms\n", elapsedTime);  
//
//  long histoCount = 0;  
//  for (int i=0; i<256; i++) {  
//    histoCount += histo[i];  
//  }  
//  printf("Histogram Sum:  %ld\n", histoCount);  
//
//  //verify that we have the same counts via CPU  
//  for (int i = 0; i < SIZE; i++)  
//    histo[buffer[i]]--;  
//  for (int i = 0; i < 256; i++) {  
//    if (histo[i] != 0)  
//      printf("Failure at %d!\n", i);  
//  }  
//
//  HANDLE_ERROR(cudaEventDestroy(start));  
//  HANDLE_ERROR(cudaEventDestroy(stop));  
//  cudaFree(dev_histo);  
//  cudaFree(dev_buffer);  
//  free(buffer);  
//  return 0;  
//}  

int main(int argc, char* argv[])
{
  //test1();
  //test2();
  //test3();
  //test5();
  //test6();
  test8();
  //test16();
  cout<<"ok!"<<endl;
}
