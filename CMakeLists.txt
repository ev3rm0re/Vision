cmake_minimum_required(VERSION 3.5.0)
project(armor_detect VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_LIB $ENV{OpenCV_DIR}/build/x64/vc16/lib/opencv_world490d.lib)
    set(OpenVINO_LIB $ENV{OpenVINO_DIR}/runtime/lib/intel64/Debug/openvinod.lib)
else()
    set(OpenCV_LIB $ENV{OpenCV_DIR}/build/x64/vc16/lib/opencv_world490.lib)
    set(OpenVINO_LIB $ENV{OpenVINO_DIR}/runtime/lib/intel64/Release/openvino.lib)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include
    $ENV{OpenCV_DIR}/build/include
    $ENV{OpenVINO_DIR}/runtime/include
    $ENV{MVCAM_COMMON_RUNENV}/Includes
)

link_libraries(
    ${OpenCV_LIB}
    ${OpenVINO_LIB}
    $ENV{MVCAM_COMMON_RUNENV}/Libraries/win64/MvCameraControl.lib
)

add_executable(armor_detect main.cpp src/camera.cpp)

target_link_libraries(armor_detect
    ${OpenCV_LIB}
    ${OpenVINO_LIB}
    $ENV{MVCAM_COMMON_RUNENV}/Libraries/win64/MvCameraControl.lib
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
