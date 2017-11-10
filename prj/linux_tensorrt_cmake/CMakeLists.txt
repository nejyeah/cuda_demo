# CMake file for TensorRt_Test

# 设定依赖的CMake版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
# 指定项目名称
PROJECT(TensorRT_Test)

# 搜索指定的OpenCV外部依赖库头文件和库文件位置
FIND_PACKAGE(OpenCV 2)
# 搜索指定的CUDA外部依赖库头文件和库文件位置
FIND_PACKAGE(CUDA)

# 打印相关信息, CMAKE_CURRENT_SOURCE_DIR指的是当前处理的CMakeLists.txt所在的路径
MESSAGE(STATUS "current path: ${CMAKE_CURRENT_SOURCE_DIR}")

# 定义用户自定义变量
SET(PATH_CPP_FILES ./../../demo/TensorRT_Test)
#MESSAGE(STATUS "cpp files path: ${PATH_CPP_FILES}")
SET(PATH_CU_FILES ./../../demo/TensorRT_Test)

# 递归查询所有匹配的文件:*.cpp和*.cu
FILE(GLOB_RECURSE CPP_LIST ${PATH_CPP_FILES}/*.cpp)
FILE(GLOB_RECURSE CU_LIST ${PATH_CU_FILES}/*.cu)
#MESSAGE(STATUS "cpp list: ${CPP_LIST}")
#MESSAGE(STATUS "cu list: ${CU_LIST}")

# 使CMake支持C++11特性
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

# 指定CUDA头文件的搜索路径
INCLUDE_DIRECTORIES(/usr/local/cuda-8.0/include)
# 指定TensorRT头文件的搜索路径
INCLUDE_DIRECTORIES(/usr/local/tensorrt/tensorrt2.1_v6.0/include)
# 指定cudnn头文件的搜索路径
INCLUDE_DIRECTORIES(/usr/local/cudnn/cudnn8.0_v6.0/include)
# 指定OpenCV头文件搜索路径
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
#MESSAGE(STATUS "opencv include directory: ${OpenCV_INCLUDE_DIRS}")

# 指定TensorRT动态库搜索路径
LINK_DIRECTORIES(/usr/local/tensorrt/tensorrt2.1_v6.0/lib)
# 指定cudnn动态库搜索路径
LINK_DIRECTORIES(/usr/local/cudnn/cudnn8.0_v6.0/lib64)

# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)

# 编译可执行程序，即生成可执行文件TensorRT_Test
#ADD_EXECUTABLE(TensorRT_Test ${CPP_LIST} ${CU_LIST}) # 不包括编译*.cu文件，需要使用CUDA_ADD_EXECUTABLE命令
CUDA_ADD_EXECUTABLE(TensorRT_Test ${CPP_LIST} ${CU_LIST})
# 用来为target添加需要链接的共享库，指定工程所用的依赖库，包括动态库和静态库
TARGET_LINK_LIBRARIES(TensorRT_Test ${OpenCV_LIBS} nvcaffe_parser nvinfer cudnn)
#MESSAGE(STATUS "opencv libs: ${OpenCV_LIBS}")

MESSAGE(STATUS "**************************************")
