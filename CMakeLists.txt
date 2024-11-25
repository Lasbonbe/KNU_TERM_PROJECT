cmake_minimum_required(VERSION 3.29)
project(KNU_TERM_PROJECT)

set(CMAKE_CXX_STANDARD 98)

set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
add_executable(KNU_TERM_PROJECT main.cpp)
target_link_libraries(KNU_TERM_PROJECT ${OpenCV_LIBS})