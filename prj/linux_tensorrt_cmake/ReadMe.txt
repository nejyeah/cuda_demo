在Linux下通过CMake编译TensorRT_Test中的测试代码步骤：
1. 将终端定位到CUDA_Test/prj/linux_tensorrt_cmake，依次执行如下命令：
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make (生成TensorRT_Test执行文件)
	$ ln -s ../../../test_data/models  ./ (将models目录软链接到build目录下)
	$ ln -s ../../../test_data/images  ./ (将images目录软链接到build目录下)
	$ ./TensorRT_Test
2. 对于有需要用OpenCV参与的读取图像的操作，需要先将对应文件中的图像路径修改为Linux支持的路径格式
