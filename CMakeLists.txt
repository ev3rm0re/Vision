cmake_minimum_required(VERSION 3.5.0)
project(Vision_WinVersion VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories("D:/Vision/opencv-4.9.0/build/install/include")
include_directories("D:/Vision/openvino/runtime/include")
include_directories("D:/Program Files/MVS/Development/Includes")

link_directories("D:/Vision/opencv-4.9.0/build/install/x64/mingw/lib")
link_directories("D:/Vision/openvino/runtime/bin/intel64/Release")
link_directories("D:/Program Files/MVS/Development/Libraries/win64")

link_libraries("D:/Vision/opencv-4.9.0/build/install/x64/mingw/lib/libopencv_world490.dll.a")
link_libraries("D:/Vision/openvino/runtime/bin/intel64/Release/openvino.dll")
link_libraries("D:/Program Files/MVS/Development/Libraries/win64/MvCameraControl.lib")

add_executable(Vision_WinVersion main.cpp ./src/camera.cpp)

target_link_libraries(Vision_WinVersion "D:/Vision/opencv-4.9.0/build/install/x64/mingw/lib/libopencv_world490.dll.a")
target_link_libraries(Vision_WinVersion "D:/Vision/openvino/runtime/bin/intel64/Release/openvino.dll")
target_link_libraries(Vision_WinVersion "D:/Program Files/MVS/Development/Libraries/win64/MvCameraControl.lib")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
