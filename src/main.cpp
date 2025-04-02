#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <camera.hpp>
#include <detector.hpp>
#include <number_classifier.hpp>
#include <serial_thread.hpp>
#include <packet.hpp>
#include <crc.hpp>
#include <ekf.hpp>

#include <chrono>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace auto_aim;

std::atomic<bool> detect_color = false; // true for red, false for blue
std::map<int, Armor> tracked_armors;
std::map<int, std::vector<cv::Point2f>> id_trajectory;

int generate_new_id() {
    static int current_id = 0;
    return current_id++;  // Generate unique ID
}

void detect() {
    std::atomic<bool> serial_ready{false};
    std::mutex color_mutex;

    // 读取配置文件
    YAML::Node params = YAML::LoadFile("../config/launch_params.yaml");

    std::string serial_port = params["serial"]["port"].as<std::string>();
    int baudrate = params["serial"]["baudrate"].as<int>();

    std::string yolo_xml_path = params["yolo"]["xml_path"].as<std::string>();
    std::string yolo_bin_path = params["yolo"]["bin_path"].as<std::string>();

    std::string camera_matrix_path = params["camera"]["matrix_path"].as<std::string>();

    std::string number_classifier_model_path = params["number_classifier"]["model_path"].as<std::string>();
    std::string number_classifier_label_path = params["number_classifier"]["label_path"].as<std::string>();



    // 初始化串口线程
    SerialThread serial_thread(serial_port, baudrate);
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
    HIK::Camera camera;

    // 根据相机内参和畸变参数实例化PnP解算器
    while (access(camera_matrix_path.c_str(), F_OK) == -1) {
        cerr << "相机内参和畸变参数文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    YAML::Node config = YAML::LoadFile(camera_matrix_path);
    vector<float> camera_vector = config["Camera matrix"].as<vector<float>>();
    vector<float> distortion_coefficients_vector = config["Distortion coefficients"].as<vector<float>>();

    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, camera_vector.data());
    cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32F, distortion_coefficients_vector.data());

    PnPSolver pnp_solver(camera_matrix, distortion_coefficients);

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
    Armor last_armor;
    Position last_position;

    EKFTracker ekf;
    bool ekf_initialized = false;

    while (true) {
        auto start = chrono::high_resolution_clock::now();
        if (camera.cap(&frame) != true) {
            camera.close();
            cerr << "Failed to capture frame" << endl;
            cerr << "reopening camera" << endl;
            while (!camera.open())
            {
                cerr << "Failed to reopen camera" << endl;
                sleep(1);
            }
        }
        
        if (frame.empty()) break;
        ov::Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);

        sort(armors.begin(), armors.end(), [](const Armor& a, const Armor& b) {
            return a.z < b.z;
        });

        if (!armors.empty()) {
            Armor armor = armors[0];

            if (armor.color == (detect_color.load() ? "red" : "blue")) {
                Position position = pnp_solver.solve(armor, frame);

                if (!ekf_initialized) {
                    ekf.init(armor.center.x, armor.center.y, armor.z, position.yaw, position.pitch, 1.0 / 80.0); // 假设 dt=0.1s
                    ekf_initialized = true;
                } else {
                    ekf.predict(1.0 / 80.0);
                    ekf.update(armor.center.x, armor.center.y, armor.z, position.yaw, position.pitch);
                }
        
                cv::Mat ekf_state = ekf.getState();
                double filtered_x = ekf_state.at<double>(0, 0);
                double filtered_y = ekf_state.at<double>(1, 0);
                double filtered_z = ekf_state.at<double>(2, 0);
                double filtered_yaw = ekf_state.at<double>(6, 0);
                double filtered_pitch = ekf_state.at<double>(8, 0);

                cv::Point2f current_point(filtered_x, filtered_y);
                // 如果轨迹非空，则获取最后一个轨迹点
                if (!id_trajectory[armor.id].empty()) {
                    cv::Point2f last_traj_point = id_trajectory[armor.id].back();
                    if (cv::norm(last_traj_point - current_point) > 20) { 
                        int old_id = armor.id;
                        int new_id = generate_new_id();
                        armor.id = new_id;
                        // 删除旧的轨迹并初始化新轨迹
                        id_trajectory.erase(old_id);
                        id_trajectory[new_id] = std::vector<cv::Point2f>();
                        // 更新 tracked_armors 中的记录
                        tracked_armors.erase(old_id);
                        tracked_armors[new_id] = armor;
                        cout << "Armor ID updated from " << old_id << " to " << new_id << endl;
                    } else {
                        // 无跳变时更新 tracked 信息
                        tracked_armors[armor.id] = armor;
                    }
                } else {
                    // 如果轨迹为空，直接更新 tracked 信息
                    tracked_armors[armor.id] = armor;
                }

                // 将当前 EKF 预测点添加到对应ID的轨迹中
                if (id_trajectory.find(armor.id) == id_trajectory.end()) {
                    id_trajectory[armor.id] = std::vector<cv::Point2f>();
                }
                id_trajectory[armor.id].push_back(current_point);
                if (id_trajectory[armor.id].size() > 500) {
                    id_trajectory[armor.id].erase(id_trajectory[armor.id].begin());
                }
                
                // 绘制当前目标轨迹（只绘制当前ID的轨迹）
                auto& trajectory = id_trajectory[armor.id];
                for (size_t i = 1; i < trajectory.size(); i++) {
                    cv::line(frame, trajectory[i - 1], trajectory[i], cv::Scalar(0, 255, 0), 2);
                }

                // 绘制当前位置、箭头及其他信息
                cv::circle(frame, cv::Point(filtered_x, filtered_y), 5, cv::Scalar(0, 0, 255), -1);
                double vx = ekf_state.at<double>(3, 0);
                double vy = ekf_state.at<double>(4, 0);
                double angle = atan2(vy, vx);
                cv::Point2f arrow_end(filtered_x + 100 * cos(angle), filtered_y + 100 * sin(angle));
                cv::arrowedLine(frame, cv::Point(filtered_x, filtered_y), arrow_end, cv::Scalar(255, 255, 255), 2);
                cv::putText(frame, "Tracking Target", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

                // 显示调试信息
                cv::putText(frame, "Tracking Target", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

                double view_angle = abs(position.yaw - last_position.yaw);
                double rotation_angle = abs(armor.yaw - last_armor.yaw);

                if (last_armor.confidence != 0.0 && last_position.yaw != 0.0
                    && rotation_angle < 0.4 && rotation_angle > 0.25) {
                    double last_d = sqrt(pow(last_armor.x, 2) + pow(last_armor.z, 2));
                    double current_d = sqrt(pow(armor.x, 2) + pow(armor.z, 2));

                    double rotation_radius = sqrt((pow(last_d, 2) + pow(current_d, 2) - 2 * last_d * current_d * cos(view_angle)) / (2 - 2 * cos(rotation_angle)));
                    double camera_to_center = sqrt(pow(position.distance, 2) + pow(rotation_radius, 2) - 2 * position.distance * rotation_radius * cos(CV_PI - armor.yaw - position.yaw));
                    camera_to_center = sqrt(pow(camera_to_center, 2) - pow(armor.y, 2));
                    double error = (camera_to_center - position.distance) / position.distance;
                    if (error < 0.1) {
                        std::cout << "rotation angle: " << rotation_angle << " rotation radius: " << rotation_radius << " camera to center: " << camera_to_center << std::endl;
                    }
                }

                line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
                line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
                putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                putText(frame, "distance: " + to_string(position.distance).substr(0, 4) + "M", armor.right_light.bottom + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

                if (serial_ready) {
                    send_packet.yaw = position.yaw;
                    send_packet.pitch = position.pitch;
                    send_packet.distance = position.distance;
                        
                    send_packet.tracking = false;
                    send_packet.id = id_unit8_map.at(armor.number);
                    send_packet.armors_num = 4;
                    send_packet.reserved = 0;
                    serial_thread.send_packet(send_packet);
                }
                last_armor = armor;
                last_position = position;
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imwrite("frame.jpg", frame);
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