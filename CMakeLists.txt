cmake_minimum_required(VERSION 3.30)
set(CMAKE_TOOLCHAIN_FILE "D:/Software/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
project(RBM LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 可执行文件
add_executable(RBM
        main.cpp
        DataRead/readData.cpp
        CudaNN/NNCore.cu
        CudaNN/Vector.cu
        CudaNN/Matrix.cu
        CudaNN/CudaFunctions.cu
        RBM/RBM.cpp
        RBM/RBM.h
)

# 包含目录
target_include_directories(RBM PRIVATE
        ${CMAKE_SOURCE_DIR}/lib
        ${CMAKE_SOURCE_DIR}/CudaNN
)

# CUDA 设置
set_target_properties(RBM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# nvcc 编译选项，避免 Debug 模式下 /Fd 参数错误
set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler /Od -Xcompiler /Zi -Xcompiler /FS")

# 拷贝资源文件夹
file(COPY ${CMAKE_SOURCE_DIR}/Data DESTINATION ${CMAKE_BINARY_DIR})
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(RBM PRIVATE Eigen3::Eigen)
