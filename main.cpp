#include <iostream>

#include <opencv2/opencv.hpp>
#include <openvino/core/core.hpp>
#include <camera.hpp>
#include <chrono>


int main()
{
    ov::Version version = ov::get_openvino_version();
    std::cout << "OpenVINO version: " << version.buildNumber << std::endl;
    
    HIK::Camera camera;
    camera.open();
    std::chrono::steady_clock::time_point begin, end;
    while (true)
    {
        begin = std::chrono::steady_clock::now();
        cv::Mat frame = camera.capture();
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - begin;
        cv::putText(frame, "FPS: " + std::to_string(1.0 / elapsed_seconds.count()), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}