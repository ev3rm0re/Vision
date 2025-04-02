cmake_minimum_required(VERSION 3.5.0)
project(Vision VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

find_library(OpenVINO_LIBRARIES NAMES libopenvino.so HINTS /usr/lib)
find_library(YAMLCPP_LIBRARIES NAMES libyaml-cpp.so HINTS /usr/lib)
find_library(OpenCV_LIBRARIES NAMES libopencv_world.so HINTS /usr/local/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/MVS/include)
include_directories(/usr/include/yaml-cpp)
include_directories(/usr/include/eigen3)

link_libraries(
    /opt/MVS/lib/64/libMvCameraControl.so
    /usr/local/lib/libopencv_world.so
    ${OpenVINO_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
)

aux_source_directory(src DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_directories(Vision PUBLIC /opt/MVS/lib)
target_link_libraries(Vision 
    ${OpenCV_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
