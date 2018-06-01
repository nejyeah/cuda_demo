#include "funset.h"  
#include <iostream>  
#include "book.h"  
#include "cpu_bitmap.h"  
#include "gpu_anim.h"  
  
using namespace std;  
  
int test1();//简单的两数相加  
int test2();//获取GPU设备相关属性  
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
  
int main(int argc, char* argv[])  
{  
  //test1();  
  test2();  
  cout<<"ok!"<<endl;  
  return 0;  
}  
  
int test1()  
{  
  int a = 2, b = 3, c = 0;  
  int* dev_c = NULL;  
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));  

  //尖括号表示要将一些参数传递给CUDA编译器和运行时系统  
  //尖括号中这些参数并不是传递给设备代码的参数，而是告诉运行时如何启动设备代码,  
  //传递给设备代码本身的参数是放在圆括号中传递的，就像标准的函数调用一样  
  add<<<1, 1>>>(a, b, dev_c);  
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));  

  printf("%d + %d = %d\n", a, b, c);  
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
  
    //copy the arrays 'a' and 'b' to the GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  
    //尖括号中的第一个参数表示设备在执行核函数时使用的并行线程块的数量  
    add_blockIdx<<<NUM,1>>>( dev_a, dev_b, dev_c );  
  
    //copy the array 'c' back from the GPU to the CPU  
    HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  
  
    //display the results  
    for (int i=0; i<NUM; i++) {  
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );  
    }  
  
    //free the memory allocated on the GPU  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaFree(dev_b));  
    HANDLE_ERROR(cudaFree(dev_c));  
  
    return 0;  
}  
  
int test4()  
{  
    //globals needed by the update routine  
    struct DataBlock {  
        unsigned char* dev_bitmap;  
    };  
  
    DataBlock   data;  
    CPUBitmap bitmap(DIM, DIM, &data);  
    unsigned char* dev_bitmap;  
  
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));  
    data.dev_bitmap = dev_bitmap;  
  
    //声明一个二维的线程格  
    //类型dim3表示一个三维数组，可以用于指定启动线程块的数量  
    //当用两个值来初始化dim3类型的变量时，CUDA运行时将自动把第3维的大小指定为1  
    dim3 grid(DIM, DIM);  
    kernel_julia<<<grid,1>>>(dev_bitmap);  
  
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,   
        bitmap.image_size(), cudaMemcpyDeviceToHost));  
  
    HANDLE_ERROR(cudaFree(dev_bitmap));  
  
    bitmap.display_and_exit();  
  
    return 0;  
}  
  
int test5()  
{  
    int a[NUM], b[NUM], c[NUM];  
    int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;  
  
    //在GPU上分配内存  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  
  
    //在CPU上为数组'a'和'b'赋值  
    for (int i = 0; i < NUM; i++) {  
        a[i] = i;  
        b[i] = i * i;  
    }  
  
    //将数组'a'和'b'复制到GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  
    add_threadIdx<<<1, NUM>>>(dev_a, dev_b, dev_c);  
  
    //将数组'c'从GPU复制到CPU  
    HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  
  
    //显示结果  
    for (int i = 0; i < NUM; i++) {  
        printf("%d + %d = %d\n", a[i], b[i], c[i]);  
    }  
  
    //释放在GPU分配的内存  
    cudaFree(dev_a);  
    cudaFree(dev_b);  
    cudaFree(dev_c);  
      
    return 0;  
}  
  
int test6()  
{  
    int a[NUM], b[NUM], c[NUM];  
    int *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;  
  
    //在GPU上分配内存  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  
  
    //在CPU上为数组'a'和'b'赋值  
    for (int i = 0; i < NUM; i++) {  
        a[i] = i;  
        b[i] = i * i / 10;  
    }  
  
    //将数组'a'和'b'复制到GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(int), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(int), cudaMemcpyHostToDevice));  
  
    add_blockIdx_threadIdx<<<128, 128>>>(dev_a, dev_b, dev_c);  
  
    //将数组'c'从GPU复制到CPU  
    HANDLE_ERROR(cudaMemcpy(c, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost));  
  
    //验证GPU确实完成了我们要求的工作  
    bool success = true;  
    for (int i = 0; i < NUM; i++) {  
        if ((a[i] + b[i]) != c[i]) {  
            printf("error: %d + %d != %d\n", a[i], b[i], c[i]);  
            success = false;  
        }  
    }  
  
    if (success)  
        printf("we did it!\n");  
  
    //释放在GPU分配的内存  
    cudaFree(dev_a);  
    cudaFree(dev_b);  
    cudaFree(dev_c);  
  
    return 0;  
}  
  
int test7()  
{  
    DataBlock data;  
    CPUAnimBitmap bitmap(DIM, DIM, &data);  
    data.bitmap = &bitmap;  
  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));  
  
    bitmap.anim_and_exit((void(*)(void*,int))generate_frame, (void(*)(void*))cleanup);  
  
    return 0;  
}  
  
void generate_frame(DataBlock *d, int ticks)  
{  
    dim3 blocks(DIM/16, DIM/16);  
    dim3 threads(16, 16);  
    ripple_kernel<<<blocks,threads>>>(d->dev_bitmap, ticks);  
  
    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));  
}  
  
//clean up memory allocated on the GPU  
void cleanup(DataBlock *d)  
{  
    HANDLE_ERROR(cudaFree(d->dev_bitmap));   
}  
  
int test8()  
{  
    float *a, *b, c, *partial_c;  
    float *dev_a, *dev_b, *dev_partial_c;  
  
    //allocate memory on the cpu side  
    a = (float*)malloc(NUM * sizeof(float));  
    b = (float*)malloc(NUM * sizeof(float));  
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));  
  
    //allocate the memory on the GPU  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));  
  
    //fill in the host memory with data  
    for (int i = 0; i < NUM; i++) {  
        a[i] = i;  
        b[i] = i*2;  
    }  
  
    //copy the arrays 'a' and 'b' to the GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, NUM * sizeof(float), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, NUM * sizeof(float), cudaMemcpyHostToDevice));   
  
    dot_kernel<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);  
  
    //copy the array 'c' back from the GPU to the CPU  
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
  
    //free memory on the gpu side  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaFree(dev_b));  
    HANDLE_ERROR(cudaFree(dev_partial_c));  
  
    //free memory on the cpu side  
    free(a);  
    free(b);  
    free(partial_c);  
  
    return 0;  
}  
  
int test9()  
{  
    DataBlock data;  
    CPUBitmap bitmap(DIM, DIM, &data);  
    unsigned char *dev_bitmap;  
  
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));  
    data.dev_bitmap = dev_bitmap;  
  
    dim3 grids(DIM / 16, DIM / 16);  
    dim3 threads(16,16);  
    julia_kernel<<<grids, threads>>>(dev_bitmap);  
  
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));  
  
    HANDLE_ERROR(cudaFree(dev_bitmap));  
  
    bitmap.display_and_exit();  
  
    return 0;  
}  
  
int test10()  
{  
    DataBlock data;  
    //capture the start time  
    cudaEvent_t start, stop;  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    CPUBitmap bitmap(DIM, DIM, &data);  
    unsigned char *dev_bitmap;  
    Sphere *s;  
  
    //allocate memory on the GPU for the output bitmap  
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));  
    //allocate memory for the Sphere dataset  
    HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));  
  
    //allocate temp memory, initialize it, copy to memory on the GPU, then free our temp memory  
    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);  
    for (int i = 0; i < SPHERES; i++) {  
        temp_s[i].r = rnd(1.0f);  
        temp_s[i].g = rnd(1.0f);  
        temp_s[i].b = rnd(1.0f);  
        temp_s[i].x = rnd(1000.0f) - 500;  
        temp_s[i].y = rnd(1000.0f) - 500;  
        temp_s[i].z = rnd(1000.0f) - 500;  
        temp_s[i].radius = rnd(100.0f) + 20;  
    }  
  
    HANDLE_ERROR(cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));  
    free(temp_s);  
  
    //generate a bitmap from our sphere data  
    dim3 grids(DIM / 16, DIM / 16);  
    dim3 threads(16, 16);  
    RayTracing_kernel<<<grids, threads>>>(s, dev_bitmap);  
  
    //copy our bitmap back from the GPU for display  
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));  
  
    //get stop time, and display the timing results  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
    printf("Time to generate:  %3.1f ms\n", elapsedTime);  
  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    HANDLE_ERROR(cudaFree(dev_bitmap));  
    HANDLE_ERROR(cudaFree(s));  
  
    // display  
    bitmap.display_and_exit();  
  
    return 0;  
}  
  
int test11()  
{  
    DataBlock data;  
    //capture the start time  
    cudaEvent_t start, stop;  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    CPUBitmap bitmap(DIM, DIM, &data);  
    unsigned char *dev_bitmap;  
  
    //allocate memory on the GPU for the output bitmap  
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));  
  
    //allocate temp memory, initialize it, copy to constant memory on the GPU, then free temp memory  
    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);  
    for (int i = 0; i < SPHERES; i++) {  
        temp_s[i].r = rnd(1.0f);  
        temp_s[i].g = rnd(1.0f);  
        temp_s[i].b = rnd(1.0f);  
        temp_s[i].x = rnd(1000.0f) - 500;  
        temp_s[i].y = rnd(1000.0f) - 500;  
        temp_s[i].z = rnd(1000.0f) - 500;  
        temp_s[i].radius = rnd(100.0f) + 20;  
    }  
  
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));  
    free(temp_s);  
  
    //generate a bitmap from our sphere data  
    dim3 grids(DIM / 16, DIM / 16);  
    dim3 threads(16, 16);  
    RayTracing_kernel<<<grids, threads>>>(dev_bitmap);  
  
    //copy our bitmap back from the GPU for display  
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));  
  
    //get stop time, and display the timing results  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
    printf("Time to generate:  %3.1f ms\n", elapsedTime);  
  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    HANDLE_ERROR(cudaFree(dev_bitmap));  
  
    //display  
    bitmap.display_and_exit();  
  
    return 0;  
}  
  
int test12()  
{  
    Heat_DataBlock data;  
    CPUAnimBitmap bitmap(DIM, DIM, &data);  
    data.bitmap = &bitmap;  
    data.totalTime = 0;  
    data.frames = 0;  
  
    HANDLE_ERROR(cudaEventCreate(&data.start));  
    HANDLE_ERROR(cudaEventCreate(&data.stop));  
  
    int imageSize = bitmap.image_size();  
  
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, imageSize));  
  
    //assume float == 4 chars in size (ie rgba)  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));  
  
    HANDLE_ERROR(cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, imageSize));  
    HANDLE_ERROR(cudaBindTexture(NULL, texIn, data.dev_inSrc, imageSize));  
    HANDLE_ERROR(cudaBindTexture(NULL, texOut, data.dev_outSrc, imageSize));  
  
    //intialize the constant data  
    float *temp = (float*)malloc(imageSize);  
  
    for (int i = 0; i < DIM*DIM; i++) {  
        temp[i] = 0;  
        int x = i % DIM;  
        int y = i / DIM;  
        if ((x>300) && (x<600) && (y>310) && (y<601))  
            temp[i] = MAX_TEMP;  
    }  
  
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;  
    temp[DIM * 700 + 100] = MIN_TEMP;  
    temp[DIM * 300 + 300] = MIN_TEMP;  
    temp[DIM * 200 + 700] = MIN_TEMP;  
  
    for (int y = 800; y < 900; y++) {  
        for (int x = 400; x < 500; x++) {  
            temp[x + y * DIM] = MIN_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice));      
  
    //initialize the input data  
    for (int y = 800; y < DIM; y++) {  
        for (int x = 0; x < 200; x++) {  
            temp[x+y*DIM] = MAX_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp,imageSize, cudaMemcpyHostToDevice));  
    free(temp);  
  
    bitmap.anim_and_exit((void (*)(void*,int))Heat_anim_gpu, (void (*)(void*))Heat_anim_exit);  
  
    return 0;  
}  
  
int test13()  
{  
    Heat_DataBlock data;  
    CPUAnimBitmap bitmap(DIM, DIM, &data);  
    data.bitmap = &bitmap;  
    data.totalTime = 0;  
    data.frames = 0;  
    HANDLE_ERROR(cudaEventCreate(&data.start));  
    HANDLE_ERROR(cudaEventCreate(&data.stop));  
  
    int imageSize = bitmap.image_size();  
  
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, imageSize));  
  
    //assume float == 4 chars in size (ie rgba)  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));  
  
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();  
    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc2, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM));  
    HANDLE_ERROR(cudaBindTexture2D(NULL, texIn2, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM));  
    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut2, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM));  
  
    //initialize the constant data  
    float *temp = (float*)malloc(imageSize);  
    for (int i = 0; i < DIM*DIM; i++) {  
        temp[i] = 0;  
        int x = i % DIM;  
        int y = i / DIM;  
        if ((x > 300) && ( x < 600) && (y > 310) && (y < 601))  
            temp[i] = MAX_TEMP;  
    }  
  
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;  
    temp[DIM * 700 + 100] = MIN_TEMP;  
    temp[DIM * 300 + 300] = MIN_TEMP;  
    temp[DIM * 200 + 700] = MIN_TEMP;  
  
    for (int y = 800; y < 900; y++) {  
        for (int x = 400; x < 500; x++) {  
            temp[x + y * DIM] = MIN_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice));      
  
    //initialize the input data  
    for (int y = 800; y < DIM; y++) {  
        for (int x = 0; x < 200; x++) {  
            temp[x + y * DIM] = MAX_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp,imageSize, cudaMemcpyHostToDevice));  
    free(temp);  
  
    bitmap.anim_and_exit((void (*)(void*,int))anim_gpu, (void (*)(void*))anim_exit);  
  
    return 0;  
}  
  
void Heat_anim_gpu(Heat_DataBlock *d, int ticks)  
{  
    HANDLE_ERROR(cudaEventRecord(d->start, 0));  
  
    dim3 blocks(DIM / 16, DIM / 16);  
    dim3 threads(16, 16);  
    CPUAnimBitmap *bitmap = d->bitmap;  
  
    //since tex is global and bound, we have to use a flag to  
    //select which is in/out per iteration  
    volatile bool dstOut = true;  
  
    for (int i = 0; i < 90; i++) {  
        float *in, *out;  
        if (dstOut) {  
            in  = d->dev_inSrc;  
            out = d->dev_outSrc;  
        } else {  
            out = d->dev_inSrc;  
            in  = d->dev_outSrc;  
        }  
  
        Heat_copy_const_kernel<<<blocks, threads>>>(in);  
        Heat_blend_kernel<<<blocks, threads>>>(out, dstOut);  
        dstOut = !dstOut;  
    }  
  
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);  
  
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));  
  
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(d->stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));  
    d->totalTime += elapsedTime;  
    ++d->frames;  
  
    printf( "Average Time per frame:  %3.1f ms\n", d->totalTime/d->frames );  
}  
  
void anim_gpu(Heat_DataBlock *d, int ticks)  
{  
    HANDLE_ERROR(cudaEventRecord(d->start, 0));  
    dim3 blocks(DIM / 16, DIM / 16);  
    dim3 threads(16, 16);  
    CPUAnimBitmap  *bitmap = d->bitmap;  
  
    //since tex is global and bound, we have to use a flag to  
    //select which is in/out per iteration  
    volatile bool dstOut = true;  
    for (int i = 0; i < 90; i++) {  
        float *in, *out;  
        if (dstOut) {  
            in  = d->dev_inSrc;  
            out = d->dev_outSrc;  
        } else {  
            out = d->dev_inSrc;  
            in  = d->dev_outSrc;  
        }  
        copy_const_kernel<<<blocks, threads>>>(in);  
        blend_kernel<<<blocks, threads>>>(out, dstOut);  
        dstOut = !dstOut;  
    }  
  
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);  
  
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));  
  
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(d->stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));  
    d->totalTime += elapsedTime;  
    ++d->frames;  
    printf("Average Time per frame:  %3.1f ms\n", d->totalTime/d->frames);  
}  
  
void Heat_anim_exit(Heat_DataBlock *d)  
{  
    cudaUnbindTexture(texIn);  
    cudaUnbindTexture(texOut);  
    cudaUnbindTexture(texConstSrc);  
  
    HANDLE_ERROR(cudaFree(d->dev_inSrc));  
    HANDLE_ERROR(cudaFree(d->dev_outSrc));  
    HANDLE_ERROR(cudaFree(d->dev_constSrc));  
  
    HANDLE_ERROR(cudaEventDestroy(d->start));  
    HANDLE_ERROR(cudaEventDestroy(d->stop));  
}  
  
//clean up memory allocated on the GPU  
void anim_exit(Heat_DataBlock *d)   
{  
    cudaUnbindTexture(texIn2);  
    cudaUnbindTexture(texOut2);  
    cudaUnbindTexture(texConstSrc2);  
    HANDLE_ERROR(cudaFree(d->dev_inSrc));  
    HANDLE_ERROR(cudaFree(d->dev_outSrc));  
    HANDLE_ERROR(cudaFree(d->dev_constSrc));  
  
    HANDLE_ERROR(cudaEventDestroy(d->start));  
    HANDLE_ERROR(cudaEventDestroy(d->stop));  
}  
  
int test14()  
{  
    GPUAnimBitmap  bitmap(DIM, DIM, NULL);  
  
    bitmap.anim_and_exit((void (*)(uchar4*, void*, int))generate_frame_opengl, NULL);  
  
    return 0;  
}  
  
int test15()  
{  
    DataBlock_opengl data;  
    GPUAnimBitmap bitmap(DIM, DIM, &data);  
    data.totalTime = 0;  
    data.frames = 0;  
    HANDLE_ERROR(cudaEventCreate(&data.start));  
    HANDLE_ERROR(cudaEventCreate(&data.stop));  
  
    int imageSize = bitmap.image_size();  
  
    //assume float == 4 chars in size (ie rgba)  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));  
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));  
  
    HANDLE_ERROR(cudaBindTexture(NULL, texConstSrc ,data.dev_constSrc, imageSize));  
    HANDLE_ERROR(cudaBindTexture(NULL, texIn, data.dev_inSrc, imageSize));  
    HANDLE_ERROR(cudaBindTexture(NULL, texOut, data.dev_outSrc, imageSize));  
  
    //intialize the constant data  
    float *temp = (float*)malloc(imageSize);  
    for (int i = 0; i < DIM*DIM; i++) {  
        temp[i] = 0;  
        int x = i % DIM;  
        int y = i / DIM;  
        if ((x>300) && (x<600) && (y>310) && (y<601))  
            temp[i] = MAX_TEMP;  
    }  
  
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;  
    temp[DIM*700+100] = MIN_TEMP;  
    temp[DIM*300+300] = MIN_TEMP;  
    temp[DIM*200+700] = MIN_TEMP;  
  
    for (int y = 800; y < 900; y++) {  
        for (int x = 400; x < 500; x++) {  
            temp[x+y*DIM] = MIN_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice));      
  
    //initialize the input data  
    for (int y = 800; y < DIM; y++) {  
        for (int x = 0; x < 200; x++) {  
            temp[x+y*DIM] = MAX_TEMP;  
        }  
    }  
  
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice));  
    free(temp);  
  
    bitmap.anim_and_exit((void (*)(uchar4*, void*, int))anim_gpu_opengl, (void (*)(void*))anim_exit_opengl);  
  
    return 0;  
}  
  
void anim_gpu_opengl(uchar4* outputBitmap, DataBlock_opengl *d, int ticks)  
{  
    HANDLE_ERROR(cudaEventRecord(d->start, 0));  
    dim3 blocks(DIM / 16, DIM / 16);  
    dim3 threads(16, 16);  
  
    //since tex is global and bound, we have to use a flag to select which is in/out per iteration  
    volatile bool dstOut = true;  
    for (int i = 0; i < 90; i++) {  
        float *in, *out;  
        if (dstOut) {  
            in  = d->dev_inSrc;  
            out = d->dev_outSrc;  
        } else {  
            out = d->dev_inSrc;  
            in  = d->dev_outSrc;  
        }  
        Heat_copy_const_kernel_opengl<<<blocks, threads>>>(in);  
        Heat_blend_kernel_opengl<<<blocks, threads>>>(out, dstOut);  
        dstOut = !dstOut;  
    }  
  
    float_to_color<<<blocks, threads>>>(outputBitmap, d->dev_inSrc);  
  
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(d->stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));  
    d->totalTime += elapsedTime;  
    ++d->frames;  
    printf("Average Time per frame:  %3.1f ms\n", d->totalTime/d->frames);  
}  
  
void anim_exit_opengl(DataBlock_opengl *d)  
{  
    HANDLE_ERROR(cudaUnbindTexture(texIn));  
    HANDLE_ERROR(cudaUnbindTexture(texOut));  
    HANDLE_ERROR(cudaUnbindTexture(texConstSrc));  
    HANDLE_ERROR(cudaFree(d->dev_inSrc));  
    HANDLE_ERROR(cudaFree(d->dev_outSrc));  
    HANDLE_ERROR(cudaFree(d->dev_constSrc));  
  
    HANDLE_ERROR(cudaEventDestroy(d->start));  
    HANDLE_ERROR(cudaEventDestroy(d->stop));  
}  
  
int test16()  
{  
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);  
  
    //capture the start time starting the timer here so that we include the cost of  
    //all of the operations on the GPU.  if the data were already on the GPU and we just   
    //timed the kernel the timing would drop from 74 ms to 15 ms.  Very fast.  
    cudaEvent_t start, stop;  
    HANDLE_ERROR( cudaEventCreate( &start ) );  
    HANDLE_ERROR( cudaEventCreate( &stop ) );  
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );  
  
    // allocate memory on the GPU for the file's data  
    unsigned char *dev_buffer;  
    unsigned int *dev_histo;  
    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));  
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));  
  
    HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(int)));  
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));  
  
    //kernel launch - 2x the number of mps gave best timing  
    cudaDeviceProp prop;  
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));  
    int blocks = prop.multiProcessorCount;  
    histo_kernel<<<blocks*2, 256>>>(dev_buffer, SIZE, dev_histo);  
  
    unsigned int histo[256];  
    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));  
  
    //get stop time, and display the timing results  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    float elapsedTime;  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
    printf("Time to generate:  %3.1f ms\n", elapsedTime);  
  
    long histoCount = 0;  
    for (int i=0; i<256; i++) {  
        histoCount += histo[i];  
    }  
    printf("Histogram Sum:  %ld\n", histoCount);  
  
    //verify that we have the same counts via CPU  
    for (int i = 0; i < SIZE; i++)  
        histo[buffer[i]]--;  
    for (int i = 0; i < 256; i++) {  
        if (histo[i] != 0)  
            printf("Failure at %d!\n", i);  
    }  
  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
    cudaFree(dev_histo);  
    cudaFree(dev_buffer);  
    free(buffer);  
  
    return 0;  
}  
  
float cuda_malloc_test(int size, bool up)  
{  
    cudaEvent_t start, stop;  
    int *a, *dev_a;  
    float elapsedTime;  
  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    a = (int*)malloc(size * sizeof(*a));  
    HANDLE_NULL(a);  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a,size * sizeof(*dev_a)));  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    for (int i=0; i<100; i++) {  
        if (up)  
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof( *dev_a ), cudaMemcpyHostToDevice));  
        else  
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));  
    }  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
  
    free(a);  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    return elapsedTime;  
}  
  
float cuda_host_alloc_test(int size, bool up)   
{  
    cudaEvent_t start, stop;  
    int *a, *dev_a;  
    float elapsedTime;  
  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    HANDLE_ERROR(cudaHostAlloc((void**)&a,size * sizeof(*a), cudaHostAllocDefault));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    for (int i=0; i<100; i++) {  
        if (up)  
            HANDLE_ERROR(cudaMemcpy(dev_a, a,size * sizeof(*a), cudaMemcpyHostToDevice));  
        else  
            HANDLE_ERROR(cudaMemcpy(a, dev_a,size * sizeof(*a), cudaMemcpyDeviceToHost));  
    }  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
  
    HANDLE_ERROR(cudaFreeHost(a));  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    return elapsedTime;  
}  
  
int test17()  
{  
    float elapsedTime;  
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;  
  
    //try it with cudaMalloc  
    elapsedTime = cuda_malloc_test(SIZE, true);  
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);  
    printf("\tMB/s during copy up:  %3.1f\n", MB/(elapsedTime/1000));  
  
    elapsedTime = cuda_malloc_test(SIZE, false);  
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);  
    printf("\tMB/s during copy down:  %3.1f\n", MB/(elapsedTime/1000));  
  
    //now try it with cudaHostAlloc  
    elapsedTime = cuda_host_alloc_test(SIZE, true);  
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);  
    printf("\tMB/s during copy up:  %3.1f\n", MB/(elapsedTime/1000));  
  
    elapsedTime = cuda_host_alloc_test(SIZE, false);  
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);  
    printf("\tMB/s during copy down:  %3.1f\n", MB/(elapsedTime/1000));  
  
    return 0;  
}  
  
int test18()  
{  
    cudaDeviceProp prop;  
    int whichDevice;  
    HANDLE_ERROR(cudaGetDevice(&whichDevice));  
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));  
  
    if (!prop.deviceOverlap) {  
        printf("Device will not handle overlaps, so no speed up from streams\n");  
        return 0;  
    }  
  
    cudaEvent_t start, stop;  
    float elapsedTime;  
    cudaStream_t stream;  
    int *host_a, *host_b, *host_c;  
    int *dev_a, *dev_b, *dev_c;  
  
    //start the timers  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    //initialize the stream  
    HANDLE_ERROR(cudaStreamCreate(&stream));  
  
    //allocate the memory on the GPU  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, NUM * sizeof(int)));  
  
    //allocate host locked memory, used to stream  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
  
    for (int i=0; i<FULL_DATA_SIZE; i++) {  
        host_a[i] = rand();  
        host_b[i] = rand();  
    }  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
    //now loop over full data, in bite-sized chunks  
    for (int i=0; i<FULL_DATA_SIZE; i+= NUM) {  
        //copy the locked memory to the device, async  
        HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a+i, NUM * sizeof(int), cudaMemcpyHostToDevice, stream));  
        HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b+i, NUM * sizeof(int), cudaMemcpyHostToDevice, stream));  
  
        singlestream_kernel<<<NUM/256, 256, 0, stream>>>(dev_a, dev_b, dev_c);  
  
        //copy the data from device to locked memory  
        HANDLE_ERROR(cudaMemcpyAsync(host_c+i, dev_c, NUM * sizeof(int), cudaMemcpyDeviceToHost, stream));  
  
    }  
  
    // copy result chunk from locked to full buffer  
    HANDLE_ERROR(cudaStreamSynchronize(stream));  
  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));  
    printf("Time taken:  %3.1f ms\n", elapsedTime);  
  
    //cleanup the streams and memory  
    HANDLE_ERROR(cudaFreeHost(host_a));  
    HANDLE_ERROR(cudaFreeHost(host_b));  
    HANDLE_ERROR(cudaFreeHost(host_c));  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaFree(dev_b));  
    HANDLE_ERROR(cudaFree(dev_c));  
    HANDLE_ERROR(cudaStreamDestroy(stream));  
  
    return 0;  
}  
  
int test19()  
{  
    cudaDeviceProp prop;  
    int whichDevice;  
    HANDLE_ERROR(cudaGetDevice(&whichDevice));  
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));  
    if (!prop.deviceOverlap) {  
        printf( "Device will not handle overlaps, so no speed up from streams\n" );  
        return 0;  
    }  
  
    //start the timers  
    cudaEvent_t start, stop;  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    //initialize the streams  
    cudaStream_t stream0, stream1;  
    HANDLE_ERROR(cudaStreamCreate(&stream0));  
    HANDLE_ERROR(cudaStreamCreate(&stream1));  
  
    int *host_a, *host_b, *host_c;  
    int *dev_a0, *dev_b0, *dev_c0;//为第0个流分配的GPU内存  
    int *dev_a1, *dev_b1, *dev_c1;//为第1个流分配的GPU内存  
  
    //allocate the memory on the GPU  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a0, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b0, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_c0, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a1, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b1, NUM * sizeof(int)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_c1, NUM * sizeof(int)));  
  
    //allocate host locked memory, used to stream  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));  
  
    for (int i=0; i<FULL_DATA_SIZE; i++) {  
        host_a[i] = rand();  
        host_b[i] = rand();  
    }  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    //now loop over full data, in bite-sized chunks  
    for (int i=0; i<FULL_DATA_SIZE; i+= NUM*2) {  
        //enqueue copies of a in stream0 and stream1  
        //将锁定内存以异步方式复制到设备上  
        HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a+i, NUM * sizeof(int), cudaMemcpyHostToDevice, stream0));  
        HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a+i+NUM, NUM * sizeof(int), cudaMemcpyHostToDevice, stream1));  
        //enqueue copies of b in stream0 and stream1  
        HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b+i, NUM * sizeof(int), cudaMemcpyHostToDevice, stream0));  
        HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b+i+NUM, NUM * sizeof(int), cudaMemcpyHostToDevice, stream1));  
  
        //enqueue kernels in stream0 and stream1     
        singlestream_kernel<<<NUM/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);  
        singlestream_kernel<<<NUM/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);  
  
        //enqueue copies of c from device to locked memory  
        HANDLE_ERROR(cudaMemcpyAsync(host_c+i, dev_c0, NUM * sizeof(int), cudaMemcpyDeviceToHost, stream0));  
        HANDLE_ERROR(cudaMemcpyAsync(host_c+i+NUM, dev_c1, NUM * sizeof(int), cudaMemcpyDeviceToHost, stream1));  
    }  
  
    float elapsedTime;  
  
    HANDLE_ERROR(cudaStreamSynchronize(stream0));  
    HANDLE_ERROR(cudaStreamSynchronize(stream1));  
  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start, stop));  
    printf( "Time taken:  %3.1f ms\n", elapsedTime );  
  
    //cleanup the streams and memory  
    HANDLE_ERROR(cudaFreeHost(host_a));  
    HANDLE_ERROR(cudaFreeHost(host_b));  
    HANDLE_ERROR(cudaFreeHost(host_c));  
    HANDLE_ERROR(cudaFree(dev_a0));  
    HANDLE_ERROR(cudaFree(dev_b0));  
    HANDLE_ERROR(cudaFree(dev_c0));  
    HANDLE_ERROR(cudaFree(dev_a1));  
    HANDLE_ERROR(cudaFree(dev_b1));  
    HANDLE_ERROR(cudaFree(dev_c1));  
    HANDLE_ERROR(cudaStreamDestroy(stream0));  
    HANDLE_ERROR(cudaStreamDestroy(stream1));  
  
    return 0;  
}  
  
float malloc_test(int size)  
{  
    cudaEvent_t start, stop;  
    float *a, *b, c, *partial_c;  
    float *dev_a, *dev_b, *dev_partial_c;  
    float elapsedTime;  
  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    //allocate memory on the CPU side  
    a = (float*)malloc(size * sizeof(float));  
    b = (float*)malloc(size * sizeof(float));  
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));  
  
    //allocate the memory on the GPU  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));  
  
    //fill in the host memory with data  
    for (int i=0; i<size; i++) {  
        a[i] = i;  
        b[i] = i * 2;  
    }  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
    //copy the arrays 'a' and 'b' to the GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));   
  
    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);  
    //copy the array 'c' back from the GPU to the CPU  
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));  
  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start, stop));  
  
    //finish up on the CPU side  
    c = 0;  
    for (int i=0; i<blocksPerGrid; i++) {  
        c += partial_c[i];  
    }  
  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaFree(dev_b));  
    HANDLE_ERROR(cudaFree(dev_partial_c));  
  
    //free memory on the CPU side  
    free(a);  
    free(b);  
    free(partial_c);  
  
    //free events  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    printf("Value calculated:  %f\n", c);  
  
    return elapsedTime;  
}  
  
float cuda_host_alloc_test(int size)  
{  
    cudaEvent_t start, stop;  
    float *a, *b, c, *partial_c;  
    float *dev_a, *dev_b, *dev_partial_c;  
    float elapsedTime;  
  
    HANDLE_ERROR(cudaEventCreate(&start));  
    HANDLE_ERROR(cudaEventCreate(&stop));  
  
    //allocate the memory on the CPU  
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size*sizeof(float), cudaHostAllocWriteCombined |cudaHostAllocMapped));  
    HANDLE_ERROR(cudaHostAlloc((void**)&b, size*sizeof(float), cudaHostAllocWriteCombined |cudaHostAllocMapped));  
    HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid*sizeof(float), cudaHostAllocMapped));  
  
    //find out the GPU pointers  
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));  
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));  
    HANDLE_ERROR( cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));  
  
    //fill in the host memory with data  
    for (int i=0; i<size; i++) {  
        a[i] = i;  
        b[i] = i*2;  
    }  
  
    HANDLE_ERROR(cudaEventRecord(start, 0));  
  
    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);  
  
    HANDLE_ERROR(cudaThreadSynchronize());  
    HANDLE_ERROR(cudaEventRecord(stop, 0));  
    HANDLE_ERROR(cudaEventSynchronize(stop));  
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start, stop));  
  
    //finish up on the CPU side  
    c = 0;  
    for (int i=0; i<blocksPerGrid; i++) {  
        c += partial_c[i];  
    }  
  
    HANDLE_ERROR(cudaFreeHost(a));  
    HANDLE_ERROR(cudaFreeHost(b));  
    HANDLE_ERROR(cudaFreeHost(partial_c));  
  
    // free events  
    HANDLE_ERROR(cudaEventDestroy(start));  
    HANDLE_ERROR(cudaEventDestroy(stop));  
  
    printf("Value calculated:  %f\n", c);  
  
    return elapsedTime;  
}  
  
int test20()  
{  
    cudaDeviceProp prop;  
    int whichDevice;  
    HANDLE_ERROR(cudaGetDevice(&whichDevice));  
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));  
    if (prop.canMapHostMemory != 1) {  
        printf( "Device can not map memory.\n" );  
        return 0;  
    }  
  
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));  
  
    //try it with malloc  
    float elapsedTime = malloc_test(NUM);  
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);  
  
    //now try it with cudaHostAlloc  
    elapsedTime = cuda_host_alloc_test(NUM);  
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);  
  
    return 0;  
}  
  
void* routine(void *pvoidData)  
{  
    DataStruct *data = (DataStruct*)pvoidData;  
    HANDLE_ERROR(cudaSetDevice(data->deviceID));  
  
    int size = data->size;  
    float *a, *b, c, *partial_c;  
    float *dev_a, *dev_b, *dev_partial_c;  
  
    //allocate memory on the CPU side  
    a = data->a;  
    b = data->b;  
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));  
  
    //allocate the memory on the GPU  
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));  
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));  
  
    //copy the arrays 'a' and 'b' to the GPU  
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size*sizeof(float), cudaMemcpyHostToDevice));  
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice));   
  
    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);  
    //copy the array 'c' back from the GPU to the CPU  
    HANDLE_ERROR(cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));  
  
    //finish up on the CPU side  
    c = 0;  
    for (int i=0; i<blocksPerGrid; i++) {  
        c += partial_c[i];  
    }  
  
    HANDLE_ERROR(cudaFree(dev_a));  
    HANDLE_ERROR(cudaFree(dev_b));  
    HANDLE_ERROR(cudaFree(dev_partial_c));  
  
    //free memory on the CPU side  
    free(partial_c);  
  
    data->returnValue = c;  
    return 0;  
}  
  
int test21()  
{  
    int deviceCount;  
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));  
    if (deviceCount < 2) {  
        printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);  
        return 0;  
    }  
  
    float *a = (float*)malloc(sizeof(float) * NUM);  
    HANDLE_NULL(a);  
    float *b = (float*)malloc(sizeof(float) * NUM);  
    HANDLE_NULL(b);  
  
    //fill in the host memory with data  
    for (int i=0; i<NUM; i++) {  
        a[i] = i;  
        b[i] = i*2;  
    }  
  
    //prepare for multithread  
    DataStruct  data[2];  
    data[0].deviceID = 0;  
    data[0].size = NUM/2;  
    data[0].a = a;  
    data[0].b = b;  
  
    data[1].deviceID = 1;  
    data[1].size = NUM/2;  
    data[1].a = a + NUM/2;  
    data[1].b = b + NUM/2;  
  
    CUTThread thread = start_thread(routine, &(data[0]));  
    routine(&(data[1]));  
    end_thread(thread);  
  
    //free memory on the CPU side  
    free(a);  
    free(b);  
  
    printf("Value calculated:  %f\n", data[0].returnValue + data[1].returnValue);  
  
    return 0;  
}  

