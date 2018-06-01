#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <iostream>  
using namespace std;  
  
const int threadsPerBlock=512 ;   
const int N=2048;  
const int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;  
  
__global__ void ReductionSum(float *d_a, float *d_partial_sum)  
{  
    //申请共享内存，存在于每个block中   
    __shared__ float partialSum[threadsPerBlock];  
  
    //确定索引  
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    int tid = threadIdx.x;  
  
    //传global memory数据到shared memory  
    partialSum[tid]=d_a[i];  
  
    //传输同步  
    __syncthreads();  
      
    //在共享存储器中进行规约  
    for(int stride = 1; stride < blockDim.x; stride*=2)  
    {  
        if(tid%(2*stride)==0) partialSum[tid]+=partialSum[tid+stride];  
        __syncthreads();  
    }  
    //将当前block的计算结果写回输出数组  
    if(tid==0)    
        d_partial_sum[blockIdx.x] = partialSum[0];  
}  
  
int main()  
{  
    //申请host端内存及初始化  
    float   *h_a,*h_partial_sum;      
    h_a = (float*)malloc( N*sizeof(float) );  
    h_partial_sum = (float*)malloc( blocksPerGrid*sizeof(float));  
      
    for (int i=0; i < N; ++i)  h_a[i] = i;  
      
    //分配显存空间  
    int size = sizeof(float);  
    float *d_a;  
    float *d_partial_sum;  
    cudaMalloc((void**)&d_a,N*size);  
    cudaMalloc((void**)&d_partial_sum,blocksPerGrid*size);  
  
    //把数据从Host传到Device  
    cudaMemcpy(d_a, h_a, size*N, cudaMemcpyHostToDevice);  
  
    //调用内核函数  
    ReductionSum<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_partial_sum);  
  
    //将结果传回到主机端  
    cudaMemcpy(h_partial_sum, d_partial_sum, size*blocksPerGrid, cudaMemcpyDeviceToHost);  
  
    //将部分和求和  
    int sum=0;  
    for (int i=0; i < blocksPerGrid; ++i)  sum += h_partial_sum[i];  
  
    cout<<"sum="<<sum<<endl;  
      
    //释放显存空间  
    cudaFree(d_a);  
    cudaFree(d_partial_sum);  
  
    free(h_a);  
    free(h_partial_sum);  
  
    return 0;  
}  
