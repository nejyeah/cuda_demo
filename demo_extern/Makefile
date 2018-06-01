all:
	nvcc --gpu-architecture=sm_61 --device-c task1.cu common_cuda.cu
	nvcc --gpu-architecture=sm_61 --device-link task1.o common_cuda.o --output-file link.o
	g++ -c common_c.cpp -o common_c.o
	g++ -c main.cpp -o main.o
	g++  main.o common_c.o task1.o  link.o common_cuda.o -o test -L/usr/local/cuda/lib64 -lcudart

clear:
	rm *.o
