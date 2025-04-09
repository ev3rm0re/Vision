#include <iostream>
#include <chrono>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <camera.hpp>
#include <detector.hpp>
#include <number_classifier.hpp>
#include <serial_thread.hpp>
#include <packet.hpp>
#include <crc.hpp>
#include <calibrate.hpp>
#include <armor_tracker.hpp>

using namespace std;
using namespace cv;

std::atomic<bool> detect_color = false; // true for red, false for blue

void detect() {
    // 读取配置文件
    std::string config_path = "../config/launch_params.yaml";
    while (access(config_path.c_str(), F_OK) == -1) {
        cerr << "配置文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    YAML::Node params = YAML::LoadFile(config_path);

    std::string serial_port = params["serial"]["port"].as<std::string>();
    int baudrate = params["serial"]["baudrate"].as<int>();

    std::string yolo_xml_path = params["yolo"]["xml_path"].as<std::string>();
    std::string yolo_bin_path = params["yolo"]["bin_path"].as<std::string>();

    std::string camera_matrix_path = params["camera"]["matrix_path"].as<std::string>();
    float exposureTime = params["camera"]["exposure_time"].as<float>();
    float gain = params["camera"]["gain"].as<float>();

    std::string number_classifier_model_path = params["number_classifier"]["model_path"].as<std::string>();
    std::string number_classifier_label_path = params["number_classifier"]["label_path"].as<std::string>();



    // 初始化串口线程
    SerialThread serial_thread(serial_port, baudrate);
    bool serial_ready = false;
    while (!serial_ready) {
        try {
            serial_thread.start();
            serial_ready = true;
            std::cout << "Serial thread started successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start serial thread: " << e.what() << ". Retrying in 1 second..." << std::endl;;
            std:: this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // 检查文件是否存在
    while (access(yolo_xml_path.c_str(), F_OK) == -1 || access(yolo_bin_path.c_str(), F_OK) == -1) {
        cerr << "YOLO模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }

    // YOLO目标检测器
    unique_ptr<YoloDet> det = make_unique<YoloDet>(yolo_xml_path, yolo_bin_path);
    // 装甲板检测器
    unique_ptr<ArmorDet> armor_det = make_unique<ArmorDet>();

    // 相机
    HIK::Camera camera(exposureTime, gain);

    // 根据相机内参和畸变参数实例化PnP解算器
    while (access(camera_matrix_path.c_str(), F_OK) == -1) {
        cerr << "相机内参和畸变参数文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    YAML::Node config = YAML::LoadFile(camera_matrix_path);
    vector<double> camera_vector = config["Camera matrix"].as<vector<double>>();
    vector<double> distortion_coefficients_vector = config["Distortion coefficients"].as<vector<double>>();

    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F, camera_vector.data());
    cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_64F, distortion_coefficients_vector.data());

    // 数字分类器
    while (access(number_classifier_model_path.c_str(), F_OK) == -1 || access(number_classifier_label_path.c_str(), F_OK) == -1) {
        cerr << "数字分类器模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    NumberClassifier nc(number_classifier_model_path, number_classifier_label_path, 0.6);

    // 串口数据包
    SendPacket send_packet;
    ReceivePacket receive_packet;
    std::map<std::string, uint8_t> id_unit8_map{
    {"negative", -1},  {"outpost", 0}, {"1", 1}, {"2", 2},
    {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

    // 用于存储图像帧
    cv::Mat frame;

    // 打开摄像头
    bool isopened = camera.open();
    int frame_count = 0;
    
    ArmorTracker armor_tracker(camera_matrix, distortion_coefficients);

    while (true) {
        auto start = chrono::high_resolution_clock::now();

        if (camera.cap(&frame) != true) {
            camera.close();
            cerr << "Failed to capture frame" << endl;
            cerr << "reopening camera" << endl;
            while (!camera.open()) {
                cerr << "Failed to reopen camera" << endl;
                sleep(1);
            }
        }
        
        if (frame.empty()) {
            std::cout << "Frame is empty, skipping..." << std::endl;
            continue;
        }
        ov::Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);

        armor_tracker.track(armors, frame, frame_count);

        for (auto& armor : armors) {
            if (armor.color == (detect_color.load() ? "red" : "blue")) {
                line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 1);
                line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 1);
                putText(frame, "armor id: " + to_string(armor.track_id), armor.right_light.top + cv::Point2f(5, -40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, "distance: " + to_string(armor.z).substr(0, 3), armor.right_light.center + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        if (frame_count < 100) cv::imwrite("../images/frame/frame" + to_string(frame_count++) + ".jpg", frame);
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27) break;
    }
    camera.close();
    cv::destroyAllWindows();
    serial_thread.stop();
}

int main(int argc, char **argv) {
    detect();
    return 0;
}