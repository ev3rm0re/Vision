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
using namespace cv;
using namespace ov;

namespace auto_aim
{
    class YoloDet
    {
    public:
        YoloDet(const string &xml_path, const string &bin_path);
        vector<Scalar> colors = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0),
                                 Scalar(255, 100, 50), Scalar(50, 100, 255), Scalar(255, 50, 100)};
        const vector<string> class_names = {"red", "blue"};
        Core core = Core();
        shared_ptr<Model> model;
        CompiledModel compiled_model;
        InferRequest infer_request;
        float scale;

        Mat letterbox(const Mat &source);

        Tensor infer(const Mat &image);
        vector<vector<int>> postprocess(const Tensor &output, const float &score_threshold, const float &iou_threshold) const;
    };

    class ArmorDet
    {
    public:
        vector<Armor> detect(const vector<vector<int>> &boxes, const Mat &image);
        vector<Light> find_lights(const Mat &roi_image, const Point2f &roi_tl);
        bool is_light(const Light &light);
        vector<Armor> match_lights(const vector<Light> &lights, vector<int> &results);
        ArmorType is_armor(const Light &light1, const Light &light2);
    };

    class PnPSolver
    {
    public:
        PnPSolver(const Mat &camera_matrix, const Mat &dist_coeffs);
        Mat camera_matrix;
        Mat dist_coeffs;
        Mat rvec;
        Mat tvec;
        Mat rmat;
        vector<Point3f> object_points;
        vector<Point2f> image_points;
        vector<double> solve(const Armor &armor);
    };

}
#endif