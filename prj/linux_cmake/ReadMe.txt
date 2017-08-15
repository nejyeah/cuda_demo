在Linux下通过CMake编译CUDA_Test中的测试代码步骤：
1. 将终端定位到CUDA_Test/prj/linux_cmake，依次执行如下命令：
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make (生成CUDA_Test执行文件)
	$ ./CUDA_Test
2. 对于有需要用OpenCV参与的读取图像的操作，需要先将对应文件中的图像路径修改为Linux支持的路径格式