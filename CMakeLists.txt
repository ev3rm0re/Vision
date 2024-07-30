cmake_minimum_required(VERSION 3.5.0)
project(Vision VERSION 0.1.0 LANGUAGES C CXX)

add_definitions(-Wno-deprecated-declarations)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

find_library(YAMLCPP_LIBRARIES NAMES libyaml-cpp.so HINTS /usr/lib)

# TensorRT
# include_directories(/home/ev3rm0re/Programs/TensorRT-10.2.0.19/include)
find_library(TENSORRT_LIBRARIES NAME libnvinfer.so HINTS "/usr/lib/aarch64-linux-gnu")
find_library(TENSORRT_PLUGIN_LIBRARIES NAME libnvinfer_plugin.so HINTS "/usr/lib/aarch64-linux-gnu")
link_directories(${TENSORRT_LIBRARIES})
link_directories(${TENSORRT_PLUGIN_LIBRARIES})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/MVS/include)
include_directories(/usr/include/yaml-cpp)

link_libraries(
    /opt/MVS/lib/aarch64/libMvCameraControl.so
    ${OpenCV_LIBS}
    ${YAMLCPP_LIBRARIES}
)

# link_directories(/usr/lib)

add_executable(Vision src/main.cpp src/camera.cpp src/detector.cpp src/number_classifier.cpp src/timestamp.cpp src/stream.cpp src/serialport.cpp src/crc.cpp)

# target_link_directories(Vision PUBLIC /opt/MVS/lib)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenVINO_LIBS} ${TENSORRT_LIBRARIES} ${TENSORRT_PLUGIN_LIBRARIES} ${CUDA_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
