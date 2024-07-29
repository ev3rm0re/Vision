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

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "common.hpp"

#include "armor.hpp"

using namespace std;

namespace auto_aim
{

  class YOLOv8RT
  {
  public:
    explicit YOLOv8RT(const std::string &engine_file_path);
    ~YOLOv8RT();

    void make_pipe();
    void copy_from_Mat(const cv::Mat &image);
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
    void infer();
    void postprocess(std::vector<det::Object> &objs,
                     float score_thres = 0.25f,
                     float iou_thres = 0.65f,
                     int topk = 100,
                     int num_labels = 2);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<det::Binding> input_bindings;
    std::vector<det::Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    det::PreParam pparam;

  private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
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