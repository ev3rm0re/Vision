cmake_minimum_required(VERSION 3.15)
project(Vision VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED)

set(YAMLCPP_INCLUDE_DIRS /usr/include/yaml-cpp)
set(MVS_INCLUDE_DIRS /opt/MVS/include)
set(OpenVINO_INCLUDE_DIRS /usr/include/openvino)
set(VTK_INCLUDE_DIRS /usr/include/vtk-9.1)

find_library(OpenVINO_LIBRARIES NAMES libopenvino.so HINTS /usr/lib)
find_library(YAMLCPP_LIBRARIES NAMES libyaml-cpp.so HINTS /usr/lib)
find_library(MVS_LIBRARY NAMES libMvCameraControl.so HINTS /opt/MVS/lib/64)

aux_source_directory(src DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_include_directories(Vision PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ${MVS_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(Vision PRIVATE
    ${OpenCV_LIBRARIES}
    ${OpenVINO_LIBRARIES}
    ${MVS_LIBRARY}
    ${YAMLCPP_LIBRARIES}
    ${VTK_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
