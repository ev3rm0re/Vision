#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <camera.hpp>
#include <detector.hpp>

#include <chrono>
#include <yaml-cpp/yaml.h>

using namespace ov;
using namespace cv;
using namespace std;
using namespace auto_aim;

int main(){
    HIK::Camera camera;
    string xml_path = "/home/ev3rm0re/workspace/Vision_CmakeGcc/models/03.16_yolov8n_e50.xml";
    string bin_path = "/home/ev3rm0re/workspace/Vision_CmakeGcc/models/03.16_yolov8n_e50.bin";
    unique_ptr<YoloDet> det = make_unique<YoloDet>(xml_path, bin_path);
    unique_ptr<ArmorDet> armor_det = make_unique<ArmorDet>();

    YAML::Node config = YAML::LoadFile("/home/ev3rm0re/workspace/Vision_CmakeGcc/config/camera_matrix.yaml");
    vector<float> camera_vector = config["Camera matrix"].as<vector<float>>();
    vector<float> distortion_coefficients_vector = config["Distortion coefficients"].as<vector<float>>();
    Mat camera_matrix = Mat(3, 3, CV_32F, camera_vector.data());
    Mat distortion_coefficients = Mat(1, 5, CV_32F, distortion_coefficients_vector.data());
    PnPSolver pnp_solver(camera_matrix, distortion_coefficients);

    vector<vector<double>> datas;

    Mat frame;
    bool isopened = camera.open();

    while(isopened){
        auto start = chrono::high_resolution_clock::now();
        camera.cap(&frame);
        Tensor output = det.get()->infer(frame);
        vector<vector<int>> results = det.get()->postprocess(output, 0.3, 0.5);
        vector<Armor> armors = armor_det.get()->detect(results, frame);
        auto end = chrono::high_resolution_clock::now();
        double fps = 1e9 / chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps).substr(0, 5), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        for (Armor armor : armors) {
            datas.push_back(pnp_solver.solve(armor));
			line(frame, armor.left_light.top, armor.right_light.bottom, Scalar(0, 255, 0), 2);
            line(frame, armor.left_light.bottom, armor.right_light.top, Scalar(0, 255, 0), 2);
            putText(frame, armor.classfication_result, armor.left_light.top, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            putText(frame, to_string(armor.confidence).substr(0, 2), armor.left_light.top + Point2f(60, 0), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
		}
        // TODO: 通过串口发送数据
        imshow("frame", frame);
        if (waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}