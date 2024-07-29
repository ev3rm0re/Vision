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

using namespace std;
using namespace auto_aim;

void detect(int argc, char **argv)
{
    // 全局变量，检测的颜色
    string detect_color = "blue";

    if (argc < 6)
    {
        cerr << "Usage: " << argv[0] << " detect [yolo_engine_path] [camera_yaml] [number_classifier_onnx] [number_classifier_label_txt]" << endl;
        return;
    }

    string engine_path = argv[2];
    // 检查文件是否存在
    while (access(engine_path.c_str(), F_OK) == -1)
    {
        cerr << "YOLO模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    // YOLO目标检测器
    cudaSetDevice(0);
    unique_ptr<YOLOv8RT> yolo = make_unique<YOLOv8RT>(engine_path);
    yolo.get()->make_pipe();
    // 装甲板检测器
    unique_ptr<ArmorDet> armor_det = make_unique<ArmorDet>();

    // 实例化相机类
    HIK::Camera camera;

    // 根据相机内参和畸变参数实例化PnP解算器
    while (access(argv[3], F_OK) == -1)
    {
        cerr << "相机内参和畸变参数文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    YAML::Node config = YAML::LoadFile(argv[3]);
    // YAML::Node config = YAML::LoadFile("../config/camera_matrix.yaml");
    vector<float> camera_vector = config["Camera matrix"].as<vector<float>>();
    vector<float> distortion_coefficients_vector = config["Distortion coefficients"].as<vector<float>>();
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, camera_vector.data());
    cv::Mat distortion_coefficients = cv::Mat(1, 5, CV_32F, distortion_coefficients_vector.data());
    PnPSolver pnp_solver(camera_matrix, distortion_coefficients);

    // 数字分类器
    while (access(argv[4], F_OK) == -1 || access(argv[5], F_OK) == -1)
    {
        cerr << "数字分类器模型文件不存在，请检查文件路径" << endl;
        sleep(1);
    }
    NumberClassifier nc(argv[4], argv[5], 0.6);

    vector<Armor> armors;
    vector<vector<int>> results;
    vector<det::Object> output;
    // 用于存储pnp解算后的数据
    vector<vector<double>> datas;

    // 用于存储图像帧
    cv::Mat frame;

    // 打开摄像头
    bool isopened = camera.open();

    while (1)
    {
        armors.clear();
        results.clear();
        output.clear();
        datas.clear();
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
        // cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        yolo->copy_from_Mat(frame);
        yolo->infer();
        yolo->postprocess(output, 0.5);
        for (auto &obj : output)
        {
            std::vector<int> result;
            result.push_back(obj.rect.x);
            result.push_back(obj.rect.y);
            result.push_back(obj.rect.x + obj.rect.width);
            result.push_back(obj.rect.y + obj.rect.height);
            result.push_back(obj.label);
            results.push_back(result);
        }
        armors = armor_det.get()->detect(results, frame);
        nc.extractNumbers(frame, armors);
        nc.classify(armors);
        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (Armor armor : armors)
        {
            if (armor.color != detect_color)
                continue;
            datas.push_back(pnp_solver.solve(armor));
            armor.distance = datas.back()[4];
            line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
            line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
            putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2), armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            putText(frame, "distance: " + to_string(armor.distance).substr(0, 4) + "M", armor.right_light.bottom + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        if (datas.size() > 0)
        {
            sort(datas.begin(), datas.end(), [](vector<double> a, vector<double> b)
                 { return a[2] < b[2]; });               // 按距离从小到大排序
            datas.erase(datas.begin() + 1, datas.end()); // 只保留最近的目标 TODO: 可以手动切换目标
        }

        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    // cap.release();
    camera.close();
    cv::destroyAllWindows();
}

void calibrate()
{
    // 设置棋盘格的尺寸（内角点数）
    cv::Size boardSize(8, 6); // 在此示例中，我们使用8x6的棋盘格

    // 准备存储角点坐标的向量
    vector<vector<cv::Point3f>> objectPoints; // 世界坐标系中的3D点
    vector<vector<cv::Point2f>> imagePoints;  // 图像平面中的2D点

    // 准备棋盘格角点的3D坐标
    vector<cv::Point3f> obj;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            obj.push_back(cv::Point3f(j, i, 0));
        }
    }

    // 打开摄像头
    HIK::Camera camera;
    camera.open();

    cv::Mat frame;
    vector<cv::Point2f> corners;
    bool calibrationDone = false;

    while (!calibrationDone)
    {
        camera.cap(&frame); // 从摄像头捕获一帧图像

        // 查找棋盘格角点
        bool found = findChessboardCorners(frame, boardSize, corners);
        if (found)
        {
            cv::Mat gray;
            cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);

            // 显示角点
            drawChessboardCorners(frame, boardSize, cv::Mat(corners), found);
        }

        cv::imshow("Calibration", frame);

        // 等待按键，按下ESC键退出标定
        char key = cv::waitKey(1000);
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
    cv::Mat cameraMatrix, distCoeffs;
    vector<cv::Mat> rvecs, tvecs;
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