cmake_minimum_required(VERSION 3.10)
project(flycv_test)
add_compile_options(-std=c++11)
find_library(
    log-lib
    log)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/flycv/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/flycv/${ANDROID_ABI})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${log-lib}
    libflycv_static.a
    libpng16.a
    libturbojpeg.a
    libz.a)