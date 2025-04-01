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

#include <chrono>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace auto_aim;

std::atomic<bool> detect_color = false; // true for red, false for blue

void detect(int argc, char **argv)
{
    std::atomic<bool> serial_ready{false};
    std::mutex color_mutex;

    // 初始化串口线程
    SerialThread serial_thread("/dev/ttyUSB0", 115200);

    try {
        serial_thread.start();
        serial_ready = true;
        std::cout << "Serial thread started successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start serial thread: " << e.what() << std::endl;
        return;
    }

    if (argc < 7)
    {
        cerr << "Usage: " << argv[0] << " detect [yolo_xml_path] [yolo_bin_path] [camera_yaml] [number_classifier_onnx] [number_classifier_label_txt]" << endl;
        return;
    }
    string xml_path = argv[2];
    string bin_path = argv[3];
    // string xml_path = "../models/03.16_yolov8n_e50_int8.xml";
    // string bin_path = "../models/03.16_yolov8n_e50_int8.bin";
    // 检查文件是否存在
    while (access(xml_path.c_str(), F_OK) == -1 || access(bin_path.c_str(), F_OK) == -1)
    {
        cerr << "YOLO模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    // YOLO目标检测器
    unique_ptr<YoloDet> det = make_unique<YoloDet>(xml_path, bin_path);
    // 装甲板检测器
    unique_ptr<ArmorDet> armor_det = make_unique<ArmorDet>();

    // 实例化相机类
    HIK::Camera camera;

    // 根据相机内参和畸变参数实例化PnP解算器
    while (access(argv[4], F_OK) == -1)
    {
        cerr << "相机内参和畸变参数文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    YAML::Node config = YAML::LoadFile(argv[4]);
    // YAML::Node config = YAML::LoadFile("../config/camera_matrix.yaml");
    vector<float> camera_vector = config["Camera matrix"].as<vector<float>>();
    vector<float> distortion_coefficients_vector = config["Distortion coefficients"].as<vector<float>>();
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, camera_vector.data());
    cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32F, distortion_coefficients_vector.data());
    PnPSolver pnp_solver(camera_matrix, distortion_coefficients);

    // 数字分类器
    while (access(argv[5], F_OK) == -1 || access(argv[6], F_OK) == -1)
    {
        cerr << "数字分类器模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    NumberClassifier nc(argv[5], argv[6], 0.6);
    // NumberClassifier nc("../models/mlp.onnx", "../models/label.txt", 0.6);

    // 跟踪器
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect bbox;

    // 串口数据包
    SendPacket send_packet;
    ReceivePacket receive_packet;
    std::map<std::string, uint8_t> id_unit8_map{
    {"negative", -1},  {"outpost", 0}, {"1", 1}, {"2", 2},
    {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

    // 用于存储pnp解算后的数据
    vector<vector<double>> datas;

    // 用于存储图像帧
    cv::Mat frame;

    // 打开摄像头
    bool isopened = camera.open();

    // 用于追踪
    bool tracker_initialized = false;

    struct KFTracker {
        cv::KalmanFilter kf;
        int lost;  // 连续未匹配帧数
    };
    
    std::vector<KFTracker> kf_trackers;

    while (true)
    {
        auto start = chrono::high_resolution_clock::now();
        if (camera.cap(&frame) != true)
        {
            camera.close();
            cerr << "Failed to capture frame" << endl;
            cerr << "reopening camera" << endl;
            while (!camera.open())
            {
                cerr << "Failed to reopen camera" << endl;
                sleep(1);
            }
        }
        
        if (frame.empty())
        {
            break;
        }
        ov::Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);
        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (Armor armor : armors)
        {
            if (armor.color != (detect_color.load() ? "red" : "blue"))
                continue;
            datas.push_back(pnp_solver.solve(armor));
            armor.distance = datas.back()[4];
            line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
            line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
            putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            putText(frame, "distance: " + to_string(armor.distance).substr(0, 4) + "M", armor.right_light.bottom + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        if (datas.size() > 0)
        {
            sort(datas.begin(), datas.end(), [](vector<double> a, vector<double> b)
                 { return a[2] < b[2]; });               // 按距离从小到大排序
            datas.erase(datas.begin() + 1, datas.end()); // 只保留最近的目标

            // 定义数据关联阈值（单位：像素）
            const float association_threshold = 30.0f;

            // 对已有的每个跟踪器先进行一次预测，保存预测位置
            std::vector<cv::Point2f> predictions;
            for (size_t i = 0; i < kf_trackers.size(); i++)
            {
                cv::Mat pred = kf_trackers[i].kf.predict();
                predictions.push_back(cv::Point2f(pred.at<float>(0), pred.at<float>(1)));
            }
            // 用于标记当前跟踪器是否被更新
            std::vector<bool> kf_updated(kf_trackers.size(), false);

            // 遍历所有测量（每个 datas 中的项），尝试与已有 KF 关联
            for (const auto &data : datas)
            {
                // 平移测量
                cv::Point2f meas_pt(data[0], data[1]);
                // 旋转测量，检测到两个装甲板时计算夹角，否则取对应 KFTracker 当前的预测值
                float meas_theta = 0.f;
                if (armors.size() == 2)
                {
                    cv::Point2f center1 = armors[0].center;
                    cv::Point2f center2 = armors[1].center;
                    meas_theta = std::atan2(center2.y - center1.y, center2.x - center1.x) * 180.f / CV_PI;
                }
                else
                {
                    // 若没有双目标，此处可以选择不更新旋转信息，
                    // 或使用某个跟踪器上次预测的旋转角（例如取最接近的跟踪器）
                    // 这里简单设为0
                    meas_theta = 0.f;
                }
                cv::Mat meas = (cv::Mat_<float>(3, 1) << meas_pt.x, meas_pt.y, meas_theta);

                int best_index = -1;
                float best_dist = association_threshold;
                for (size_t i = 0; i < predictions.size(); i++)
                {
                    float dist = cv::norm(predictions[i] - meas_pt);
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_index = static_cast<int>(i);
                    }
                }
                if (best_index != -1)
                {
                    // 关联成功，更新 KFTracker
                    kf_trackers[best_index].kf.correct(meas);
                    kf_trackers[best_index].lost = 0;
                    kf_updated[best_index] = true;
                }
                else
                {
                    float dt = 0.033f;
                    // 没有与现有 KF 关联，则新建一个 KFTracker
                    cv::KalmanFilter newKF(6, 3, 0);
                    newKF.transitionMatrix = (cv::Mat_<float>(6, 6) <<
                        1, 0, dt, 0,  0,  0,
                        0, 1, 0,  dt, 0,  0,
                        0, 0, 1,  0,  0,  0,
                        0, 0, 0,  1,  0,  0,
                        0, 0, 0,  0,  1, dt,
                        0, 0, 0,  0,  0,  1);
                    newKF.measurementMatrix = (cv::Mat_<float>(3, 6) <<
                        1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 0);
                    cv::setIdentity(newKF.processNoiseCov, cv::Scalar::all(1e-4));
                    cv::setIdentity(newKF.measurementNoiseCov, cv::Scalar::all(1e-1));
                    cv::setIdentity(newKF.errorCovPost, cv::Scalar::all(1));
                    // 初始化状态：这里先用当前平移测量，同时旋转部分根据是否有双目标来决定
                    float meas_theta = 0.f;
                    if (armors.size() == 2) {
                        cv::Point2f center1 = armors[0].center;
                        cv::Point2f center2 = armors[1].center;
                        meas_theta = std::atan2(center2.y - center1.y, center2.x - center1.x) * 180.f / CV_PI;
                    } else {
                        // 无双目标时，可设为0或待后续更新
                        meas_theta = 0.f;
                    }
                    newKF.statePost.at<float>(0) = meas_pt.x;
                    newKF.statePost.at<float>(1) = meas_pt.y;
                    newKF.statePost.at<float>(2) = 0; // 初始平移速度
                    newKF.statePost.at<float>(3) = 0;
                    newKF.statePost.at<float>(4) = meas_theta;
                    newKF.statePost.at<float>(5) = 0; // 初始角速度

                    KFTracker new_tracker;
                    new_tracker.kf = newKF;
                    new_tracker.lost = 0;
                    kf_trackers.push_back(new_tracker);
                    kf_updated.push_back(true);
                }
                // 绘制测量点（红色）
                circle(frame, meas_pt, 10, cv::Scalar(0, 0, 255), 2);
            }

            // 对所有未能匹配到测量的跟踪器，增加丢失计数
            for (size_t i = 0; i < kf_trackers.size(); i++)
            {
                if (!kf_updated[i])
                {
                    kf_trackers[i].lost++;
                }
            }

            // 将连续丢失超过 10 帧的跟踪器移除
            for (auto it = kf_trackers.begin(); it != kf_trackers.end(); )
            {
                if (it->lost > 10)
                    it = kf_trackers.erase(it);
                else
                    ++it;
            }

            // 绘制所有跟踪器的预测点（蓝色）
            for (auto &tracker : kf_trackers)
            {
                cv::Mat pred = tracker.kf.predict();
                cv::Point2f predict_pt(pred.at<float>(0), pred.at<float>(1));
                circle(frame, predict_pt, 8, cv::Scalar(255, 0, 0), -1);
            }

            if (serial_ready) {
                SendPacket send_packet;
                // 使用第一个 KFTracker 预测得到的状态作为发送数据
                cv::Mat state = kf_trackers[0].kf.statePost;
                
                // 下面根据相机内参计算对应的角度偏差，作为云台的 yaw 和 pitch 值
                float target_x = state.at<float>(0);
                float target_y = state.at<float>(1);
                
                // 从 camera_matrix 获取焦距 fx, fy 和图像中心 cx, cy
                float fx = camera_matrix.at<float>(0, 0);
                float fy = camera_matrix.at<float>(1, 1);
                float cx = camera_matrix.at<float>(0, 2);
                float cy = camera_matrix.at<float>(1, 2);
                
                // 计算偏差（像素偏差转换为角度, 单位转换为度）
                float error_x = target_x - cx;
                float error_y = target_y - cy;
                
                float turretYaw = std::atan2(error_x, fx) * 180.f / CV_PI;   // 云台yaw偏差
                float turretPitch = std::atan2(error_y, fy) * 180.f / CV_PI;   // 云台pitch偏差
            
                send_packet.yaw = turretYaw;
                send_packet.pitch = turretPitch;
                // send_packet.distance 可根据需要保留，如用来控制其他逻辑
                send_packet.distance = datas[0][4];  // 或者根据实际场景修改
                
                send_packet.tracking = tracker_initialized;
                send_packet.id = id_unit8_map.at(armors[0].number);
                send_packet.armors_num = 4;
                send_packet.reserved = 0;
                serial_thread.send_packet(send_packet);
            }

            // 其他逻辑
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        datas.clear();
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    camera.close();
    cv::destroyAllWindows();
    serial_thread.stop();
}

void calibrate() {
    // 棋盘格参数
    Size boardSize(8, 5);        // 内部角点数（对应8x5个棋盘格）
    float square_size = 0.027f;
    
    // 3D坐标生成（单位：米）
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            obj.emplace_back(j * square_size, i * square_size, 0);  // [1,3](@ref)
        }
    }

    // 图像采集
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    HIK::Camera camera;
    camera.open();
    
    Mat frame;
    int count = 0;
    while (true) {
        camera.cap(&frame);
        if (frame.empty()) break;

        // 角点检测
        vector<Point2f> corners;
        bool found = findChessboardCorners(frame, boardSize, corners);
        if (found) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            // 亚像素优化（参数优化）[1](@ref)
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
            cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), criteria);
            
            objectPoints.push_back(obj);
            imagePoints.push_back(corners);
            
            // 可视化
            drawChessboardCorners(frame, boardSize, corners, found);
            putText(frame, "Captured: " + to_string(objectPoints.size()), 
                    Point(20,40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        }

        imshow("Calibration", frame);
        char key = waitKey(30);
        if (key == 27 || objectPoints.size() >= 30) break;  // 至少采集30张[1](@ref)
    }
    camera.close();

    // 标定验证
    if (objectPoints.size() < 6) {
        cerr << "Insufficient calibration images (min 6)" << endl;
        return;
    }

    // 执行标定（添加标志位优化）[1](@ref)
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    int flags = CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST;
    double rms = calibrateCamera(objectPoints, imagePoints, frame.size(), 
                                cameraMatrix, distCoeffs, rvecs, tvecs, flags);
    
    // 保存参数[1,3](@ref)
    FileStorage fs("calibration.yaml", FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix << "distortion_coefficients" << distCoeffs;
    fs.release();

    // 重投影误差计算[1,2](@ref)
    double total_error = 0;
    for (size_t i=0; i<objectPoints.size(); ++i) {
        vector<Point2f> projected_points;
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], 
                      cameraMatrix, distCoeffs, projected_points);
        double error = norm(imagePoints[i], projected_points, NORM_L2);
        total_error += error*error;
    }
    cout << "Reprojection error: " << sqrt(total_error/objectPoints.size()) 
         << " pixels" << endl;

    // 畸变校正测试[1,6](@ref)
    Mat undistorted;
    undistort(frame, undistorted, cameraMatrix, distCoeffs);
    imshow("Original vs Corrected", undistorted);
    waitKey(0);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [calibrate|detect]" << endl;
        return 1;
    }
    string mode = argv[1];
    if (mode == "calibrate")
    {
        calibrate();
    }
    else if (mode == "detect")
    {
        detect(argc, argv);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [calibrate|detect]" << endl;
        return 1;
    }
    return 0;
}