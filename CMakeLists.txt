cmake_minimum_required(VERSION 2.9)
project(faster_rcnn)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES demo.cpp faster-api.cpp faster-api.h)
add_executable(faster_rcnn ${SOURCE_FILES})

target_link_libraries(faster_rcnn ${OpenCV_LIBS} )
target_link_libraries(faster_rcnn ${Caffe_LIBRARIES})
