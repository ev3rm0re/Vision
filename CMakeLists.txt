cmake_minimum_required(VERSION 3.5.0)
project(Vision VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_library(OpenVINO_LIBRARIES NAMES libopenvino.so HINTS /usr/lib)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/MVS/include)

link_libraries(
    /opt/MVS/lib/64/libMvCameraControl.so
    /usr/local/lib/libopencv_world.so
    ${OpenVINO_LIBRARIES}
)

link_directories(/usr/lib)

add_executable(Vision main.cpp src/camera.cpp)

target_link_directories(Vision PUBLIC /opt/MVS/lib ${OpenVINO_INCLUDE_DIRS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
