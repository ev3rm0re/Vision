/*
装甲板检测的三个主要类
YoloDet: 使用OpenVINO进行装甲板位置检测
ArmorDet: 装甲板匹配检测
PnPSolver: 通过PnP解算器求解装甲板的位置
*/

#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <time.h>

#include "armor.hpp"

using namespace std;

namespace auto_aim
{
    class YoloDet
    {
    public:
        YoloDet(const string &xml_path, const string &bin_path);
        vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                 cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
        const vector<string> class_names = {"red", "blue"};
        ov::Core core = ov::Core();
        shared_ptr<ov::Model> model;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;
        float scale;

        cv::Mat letterbox(const cv::Mat &source);

        ov::Tensor infer(const cv::Mat &image);
        vector<vector<int>> postprocess(const ov::Tensor &output, const float &score_threshold, const float &iou_threshold) const;
    };

    class ArmorDet
    {
    public:
        vector<Armor> detect(const vector<vector<int>> &boxes, const cv::Mat &image);
        vector<Light> find_lights(const cv::Mat &roi_image, const cv::Point2f &roi_tl);
        bool is_light(const Light &light);
        vector<Armor> match_lights(const vector<Light> &lights, vector<int> &results);
        ArmorType is_armor(const Light &light1, const Light &light2);
    };

    class PnPSolver
    {
    public:
        PnPSolver(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs);
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        cv::Mat rvec;
        cv::Mat tvec;
        cv::Mat rmat;
        vector<cv::Point3f> object_points;
        vector<cv::Point2f> image_points;
        vector<double> solve(const Armor &armor);
    };

}
#endif