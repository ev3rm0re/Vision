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
#include <serialport.hpp>
#include <packet.hpp>
#include <crc.hpp>

#include <chrono>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

using namespace ov;
using namespace cv;
using namespace std;
using namespace auto_aim;

void detect(int argc, char **argv)
{
    // 全局变量，检测的颜色
    string detect_color = "blue";

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
    Mat camera_matrix = Mat(3, 3, CV_32F, camera_vector.data());
    Mat distortion_coefficients = Mat(1, 5, CV_32F, distortion_coefficients_vector.data());
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
    Ptr<Tracker> tracker;
    Rect bbox;

    // 打开串口
    Serial s;
    while (s.open("/dev/ttyUSB0", 115200, 8, Serial::PARITY_NONE, 1) != Serial::OK)     // 循环尝试打开串口
    {
        cerr << "Failed to open serial port" << endl;
        sleep(1);
    }

    // 串口数据包
    SendPacket send_packet;
    ReceivePacket receive_packet;
    std::map<std::string, uint8_t> id_unit8_map{
    {"negative", -1},  {"outpost", 0}, {"1", 1}, {"2", 2},
    {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

    // 用于存储pnp解算后的数据
    vector<vector<double>> datas;

    // 用于存储图像帧
    Mat frame;

    // 打开摄像头
    bool isopened = camera.open();

    // 用于追踪
    bool tracker_initialized = false;
    int id = 0;

    // 实例化一个卡尔曼滤波来预测下一帧的位置
    cv::KalmanFilter KF(6, 2, 0);

    // 初始化卡尔曼滤波器的状态转移矩阵
    KF.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, 1, 0, 0.1, 0,
                                                0, 1, 0, 1, 0, 0.1,
                                                0, 0, 1, 0, 1, 0,
                                                0, 0, 0, 1, 0, 1,
                                                0, 0, 0, 0, 1, 0,
                                                0, 0, 0, 0, 0, 1);
    // 初始化状态估计和协方差矩阵
    Mat state(6, 1, CV_32F);    // (x, y, vx, vy, ax, ay)
    Mat processNoise(6, 1, CV_32F);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    while (1)
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
        Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.3, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);
        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        for (Armor armor : armors)
        {
            if (armor.color != detect_color)
                continue;
            datas.push_back(pnp_solver.solve(armor));
            armor.distance = datas.back()[4];
            line(frame, armor.left_light.top, armor.right_light.bottom, Scalar(0, 255, 0), 2);
            line(frame, armor.left_light.bottom, armor.right_light.top, Scalar(0, 255, 0), 2);
            putText(frame, armor.classfication_result, armor.right_light.top + Point2f(5, -20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2), armor.right_light.center + Point2f(5, 0), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            putText(frame, "distance: " + to_string(armor.distance).substr(0, 4) + "M", armor.right_light.bottom + Point2f(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
        if (datas.size() > 0)
        {
            sort(datas.begin(), datas.end(), [](vector<double> a, vector<double> b)
                 { return a[2] < b[2]; });               // 按距离从小到大排序
            datas.erase(datas.begin() + 1, datas.end()); // 只保留最近的目标 TODO: 可以手动切换目标

            // 用卡尔曼滤波预测下一帧的位置
            Point2f observation(datas[0][0], datas[0][1]);
            Mat prediction = KF.predict();
            KF.correct((Mat_<float>(2, 1) << observation.x, observation.y));
            Point2f predict_pt(prediction.at<float>(0), prediction.at<float>(1));
            circle(frame, predict_pt, 8, Scalar(255, 0, 0), -1);
            circle(frame, observation, 10, Scalar(0, 0, 255), 2);

            if (!tracker_initialized)
            {
                bbox = Rect(armors[0].left_light.top, armors[0].left_light.top + Point2f(20, 20));
                tracker = TrackerCSRT::create();
                id++;
                tracker->init(frame, bbox);
                tracker_initialized = true;
            }
            else
            {
                bool tracking = tracker->update(frame, bbox);

                if (tracking)
                {
                    // rectangle(frame, bbox, Scalar(0, 255, 0), 2);
                    putText(frame, "ID: " + to_string(id), armors[0].center + Point2f(0, -40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                }
                else
                {
                    tracker.release();
                    tracker_initialized = false;
                }
            }
            // 通过串口发送数据
            send_packet.yaw = datas[0][2];              // 相机坐标系与云台坐标系相反
            send_packet.pitch = datas[0][3];
            send_packet.distance = datas[0][4];
            send_packet.tracking = tracker_initialized;
            send_packet.id = id_unit8_map.at(armors[0].number);
            send_packet.armors_num = 4;
            send_packet.reserved = 0;
            crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&send_packet), sizeof(SendPacket));
            // cout << "yaw: " << send_packet.yaw << " distance: " << send_packet.distance << " tracking: " << send_packet.tracking << " id: " << (int)send_packet.id << endl;
            if (sendPacket(s, send_packet) != sizeof(SendPacket))
            {
                cerr << "Failed to send packet, reopening serial port..." << endl;
                // 重开串口
                s.close();
                if (s.open("/dev/ttyUSB0", 115200, 8, Serial::PARITY_NONE, 1) != Serial::OK)     // 循环尝试打开串口
                {
                    cerr << "Failed to reopen serial port" << endl;
                }
            }
            // TODO: 通过串口接收数据
            receivePacket(s, receive_packet);
        }

        datas.clear();
        cv::imshow("frame", frame);
        if (waitKey(1) == 27)
        {
            break;
        }
    }
    camera.close();
    cv::destroyAllWindows();
}

void calibrate()
{
    // 设置棋盘格的尺寸（内角点数）
    Size boardSize(8, 6); // 在此示例中，我们使用8x6的棋盘格

    // 准备存储角点坐标的向量
    vector<vector<Point3f>> objectPoints; // 世界坐标系中的3D点
    vector<vector<Point2f>> imagePoints;  // 图像平面中的2D点

    // 准备棋盘格角点的3D坐标
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            obj.push_back(Point3f(j, i, 0));
        }
    }

    // 打开摄像头
    HIK::Camera camera;
    camera.open();

    Mat frame;
    vector<Point2f> corners;
    bool calibrationDone = false;

    while (!calibrationDone)
    {
        camera.cap(&frame); // 从摄像头捕获一帧图像

        // 查找棋盘格角点
        bool found = findChessboardCorners(frame, boardSize, corners);
        if (found)
        {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);

            // 显示角点
            drawChessboardCorners(frame, boardSize, Mat(corners), found);
        }

        cv::imshow("Calibration", frame);

        // 等待按键，按下ESC键退出标定
        char key = waitKey(1000);
        if (key == 27)
        {
            break;
        }
        // 标定至少使用了6个图像时退出
        if (imagePoints.size() >= 6)
        {
            calibrationDone = true;
        }
    }

    camera.close(); // 释放摄像头

    // 检查是否至少有一个图像成功找到了角点
    if (imagePoints.empty())
    {
        cerr << "No images with chessboard corners found. Exiting." << endl;
        return;
    }

    // 相机标定
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    cv::calibrateCamera(objectPoints, imagePoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    // 输出相机内参和畸变参数
    std::cout << "Camera matrix:" << endl
         << cameraMatrix << endl;
    std::cout << "Distortion coefficients:" << endl
         << distCoeffs << endl;
    return;
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