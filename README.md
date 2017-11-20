# CUDA_Test
**The main role of the project: CUDA 8.0/TensorRT's usage(each test code gives the implementation of C ++ and CUDA, respectively, and gives the calculation time for each method):**
- CUDA 8.0 test code
	- simple
		- vector add: C = A + B
		- matrix multiplication: C = A * B
		- dot product
		- Julia
		- ripple
		- green ball
		- ray tracking
		- heat conduction
		- calculate histogram
		- streams' usage
	- layer(approximate)
		- channel normalize(mean/standard deviation)
		- reverse
		- prior_vbox
- TensorRT 2.1.2 test code
	- MNIST
	- MNIST API
	- GoogleNet

**The project support platform:**
- windows10 64 bits: It can be directly build with VS2013 in windows10 64bits.
- Linux: 
	- CUDA supports cmake build(file position: prj/linux_cuda_cmake)
	- TensorRT support cmake build(file position: prj/linux_tensorrt_cmake)

**Screenshot:**  
![](https://github.com/fengbingchun/CUDA_Test/blob/master/prj/x86_x64_vc12/Screenshot.png)

**Blog:** [fengbingchun](http://blog.csdn.net/fengbingchun/article/category/1531463)
