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
#include <ekf.hpp>
#include <predictor.hpp>
#include <calibrate.hpp>

using namespace std;
using namespace cv;

std::atomic<bool> detect_color = false; // true for red, false for blue

void detect() {
    std::map<int, Armor> tracked_armors;
    std::map<int, std::vector<cv::Point3f>> id_trajectory;
    std::map<int, EKFTracker> ekf_trackers;

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

    auto last_timestamp = chrono::high_resolution_clock::now();

    const double theta = CV_PI / 4.0;

    Predictor predictor;

    int current_id = 0;
    int frame_count = 0;
    int lost_count = 0;

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
        auto timestamp = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(timestamp - last_timestamp).count() / 1e6;
        
        if (frame.empty()) {
            std::cout << "Frame is empty, skipping..." << std::endl;
            continue;
        }
        ov::Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);

        sort(armors.begin(), armors.end(), [](const Armor& a, const Armor& b) {
            return a.z < b.z;
        });

        if (!armors.empty()) {
            lost_count = 0;
            Armor armor = armors[0];

            if (armor.color == (detect_color.load() ? "red" : "blue")) {
                if (armor.id < 0) { // 假设 armor.id 默认值为 -1 或其他负数
                    armor.id = current_id; // 如果 armor.id 没有被赋值，则赋予一个新的唯一 ID
                }
                vector<cv::Mat> vec = pnp_solver.solve(armor);
                cv::drawFrameAxes(frame, camera_matrix, distortion_coefficients, vec[1], vec[0], 0.1, 1);
                predictor.getAttr(vec, armor);
                double yaw = atan2(armor.x, armor.z);
                double pitch = atan2(armor.y, armor.z);

                // 初始化或更新 EKF Tracker
                if (ekf_trackers.find(armor.id) == ekf_trackers.end()) {
                    ekf_trackers[armor.id].init(armor.x, armor.y, armor.z, yaw, pitch, 1.0 / 80.0);
                } else {
                    ekf_trackers[armor.id].predict(1.0 / 80.0);
                    ekf_trackers[armor.id].update(armor.x, armor.y, armor.z, yaw, pitch);
                }

                cv::Mat ekf_state = ekf_trackers[armor.id].getState();
                double filtered_x = ekf_state.at<double>(0, 0);
                double filtered_y = ekf_state.at<double>(1, 0);
                double filtered_z = ekf_state.at<double>(2, 0);
                double filtered_yaw = ekf_state.at<double>(6, 0);
                double filtered_pitch = ekf_state.at<double>(8, 0);

                cv::Point3f current_point(filtered_x, filtered_y, filtered_z);
                // 如果轨迹非空，则获取最后一个轨迹点
                if (!id_trajectory[armor.id].empty()) {
                    cv::Point3f last_traj_point = id_trajectory[armor.id].back();
                    double distance = cv::norm(last_traj_point - current_point);
                    double tilt_angle = abs(last_armor.yaw - armor.yaw);
                    if (distance > 0.13 || tilt_angle > 0.6 || lost_count > 4) {
                        int old_id = armor.id;
                        int new_id = ++current_id;
                        armor.id = new_id;
                        // 删除旧的轨迹并初始化新轨迹
                        id_trajectory.erase(old_id);
                        id_trajectory[new_id] = std::vector<cv::Point3f>();
                        // 更新 tracked_armors 中的记录
                        tracked_armors.erase(old_id);
                        tracked_armors[new_id] = armor;
                        // cout << ">>>>>>>Armor ID updated from " << old_id << " to " << new_id << endl;
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
                    id_trajectory[armor.id] = std::vector<cv::Point3f>();
                }
                id_trajectory[armor.id].push_back(current_point);
                if (id_trajectory[armor.id].size() > 500) {
                    id_trajectory[armor.id].erase(id_trajectory[armor.id].begin());
                }

                // // 绘制当前目标轨迹（只绘制当前ID的轨迹）
                // auto& trajectory = id_trajectory[armor.id];
                // for (size_t i = 1; i < trajectory.size(); i++) {
                //     cv::line(frame, trajectory[i - 1], trajectory[i], cv::Scalar(0, 255, 0), 2);
                // }

                // // 绘制当前位置、箭头及其他信息
                // cv::circle(frame, cv::Point(filtered_x, filtered_y), 5, cv::Scalar(0, 0, 255), -1);
                // double vx = ekf_state.at<double>(3, 0);
                // double vy = ekf_state.at<double>(4, 0);
                // double angle = atan2(vy, vx);
                // cv::Point2f arrow_end(filtered_x + 30 * cos(angle), filtered_y + 30 * sin(angle));
                // cv::arrowedLine(frame, cv::Point(filtered_x, filtered_y), arrow_end, cv::Scalar(255, 255, 255), 2);
                // cv::putText(frame, "Tracking Target", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

                if (last_armor.id == armor.id) {
                    double aim_yaw, aim_pitch;
                    bool success = predictor.calculate(last_armor, armor, duration);
                    if (success) {
                        predictor.predict(aim_yaw, aim_pitch);
                        predictor.drawPrediction(camera_matrix, frame);
                        if (serial_ready) {
                            send_packet.yaw = aim_yaw;
                            send_packet.pitch = aim_pitch;
                            
                            send_packet.tracking = false;
                            send_packet.id = id_unit8_map.at(armor.number);
                            send_packet.armors_num = 4;
                            send_packet.reserved = 0;
                            serial_thread.send_packet(send_packet);
                        }
                    }
                }
                
                line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 1);
                line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 1);
                putText(frame, "armor id: " + to_string(armor.id), armor.right_light.top + cv::Point2f(5, -40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                putText(frame, "distance: " + to_string(armor.z).substr(0, 3), armor.right_light.center + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                last_armor = armor;
                last_timestamp = timestamp;
            }
        } else {
            lost_count++;
        }

        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        if (frame_count < 20) cv::imwrite("../images/frame/frame" + to_string(frame_count++) + ".jpg", frame);
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27) break;
    }
    camera.close();
    cv::destroyAllWindows();
    serial_thread.stop();
}

int main(int argc, char **argv) {
    detect();
    // calibrate();
    return 0;
}