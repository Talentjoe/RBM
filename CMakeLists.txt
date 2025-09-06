cmake_minimum_required(VERSION 3.31)
project(RBM)

set(CMAKE_CXX_STANDARD 20)


add_executable(RBM main.cpp ./DataRead/readData.cpp ./DataRead/readData.h)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(RBM PRIVATE Eigen3::Eigen)