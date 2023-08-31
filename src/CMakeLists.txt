cmake_minimum_required(VERSION 3.4.1)

#include头文件目录
include_directories(../inc
                    /home/wangyuanwen/jdk1.8.0_231/include
                    /home/wangyuanwen/jdk1.8.0_231/include/linux
                    .)

#source directory源文件目录
file(GLOB MTCNN_SRC mtcnn.cpp
                       mtcnn_multi_jni.cpp)

set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

#添加动态连接库的路径
link_directories(../lib})

#添加ncnn库
add_library(libncnn STATIC IMPORTED )
set_target_properties(libncnn
        PROPERTIES IMPORTED_LOCATION
        ../lib/libncnn.a)

#编译为动态库
add_library(mtcnn SHARED ${MTCNN_COMPILE_CODE})

set_target_properties(mtcnn
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ../lib)

#添加工程所依赖的库
target_link_libraries( mtcnn
                       libncnn
                       )

