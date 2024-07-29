/*
目标检测器
1. YoloDet：YOLO目标检测器
2. ArmorDet：装甲板检测器
3. PnPSolver：PnP解算器
*/

#include "detector.hpp"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;

namespace auto_aim
{

    YoloDet::YoloDet(const string &xml_path, const string &bin_path)
    {
        // 初始化模型，创建推理请求
        model = core.read_model(xml_path, bin_path);
        compiled_model = core.compile_model(model, "CPU");
        infer_request = compiled_model.create_infer_request();
        scale = 0.0;
    }

    cv::Mat YoloDet::letterbox(const cv::Mat &source)
    {
        // 将图像填充为正方形
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    }

    ov::Tensor YoloDet::infer(const cv::Mat &image)
    {
        // 推理

        cv::Mat letterbox_image = YoloDet::letterbox(image);
        scale = letterbox_image.size[0] / 640.0;
        cv::Mat blob = cv::dnn::blobFromImage(letterbox_image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
        auto &input_port = compiled_model.input();
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto output = infer_request.get_output_tensor(0);
        return output;
    }

    std::vector<std::vector<int>> YoloDet::postprocess(const ov::Tensor& output, const float& score_threshold) {
        // 后处理
        float* data = output.data<float>();                         // 获取输出数据
        cv::Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, data);	// 创建输出缓冲区
        //cv::transpose(output_buffer, output_buffer);				// 转置
        std::vector<std::vector<int>> results;
        // 遍历输出层
        for (int i = 0; i < output_buffer.rows; i++) {              // 遍历每个边界框
            // 获取类别得分
            float score = output_buffer.at<float>(i, 4);
            float class_id = output_buffer.at<float>(i, 5);
            if (score > score_threshold) {							// 判断是否满足阈值
                // 获取边界框
                float ltx = output_buffer.at<float>(i, 0);
                float lty = output_buffer.at<float>(i, 1);
                float rbx = output_buffer.at<float>(i, 2);
                float rby = output_buffer.at<float>(i, 3);
                // 计算边界框真实坐标
                int left = int(ltx * scale);
                int top = int(lty * scale);
                int right = int(rbx * scale);
                int bottom = int(rby * scale);
                std::vector<int> box = { left, top, right, bottom, int(class_id), int(score) };
                // 将边界框存储
                results.push_back(box);
            }
        }
        return results;
    }

    vector<Armor> ArmorDet::detect(const vector<vector<int>> &results, const cv::Mat &image)
    {
        vector<Armor> armors;
        vector<Armor> armors_;

        // 遍历检测结果
        for (vector<int> result : results)
        {
            // 获取检测结果的ROI
            cv::Rect roi = cv::Rect(result[0], result[1], result[2] - result[0], result[3] - result[1]);
            // 筛选掉超出图像范围的ROI
            if (roi.x < 0 || roi.x + roi.width > image.cols || roi.y < 0 || roi.y + roi.height > image.rows || roi.height < 0 || roi.width < 0)
            {
                continue;
            }
            // 获取ROI图像
            cv::Mat roi_image = image(roi);
            // 获取ROI左上角坐标
            cv::Point2f roi_tl = cv::Point2f(result[0], result[1]);
            // 获取ROI中的灯条
            vector<Light> lights = find_lights(roi_image, roi_tl);
            // 匹配灯条
            vector<Armor> armors = match_lights(lights, result);
            // 筛出无效装甲板
            /*if (armor.center.x != 0 && armor.center.y != 0) {
                armors.push_back(armor);
            }*/
            armors_.insert(armors_.end(), armors.begin(), armors.end());
        }
        return armors_;
    }

    vector<Light> ArmorDet::find_lights(const cv::Mat &roi_image, const cv::Point2f &roi_tl)
    {
        vector<Light> lights;
        cv::Mat gray_image, binary_image;
        // 灰度化、高斯模糊、二值化
        cvtColor(roi_image, gray_image, cv::COLOR_BGR2GRAY);
        /*GaussianBlur(gray_image, gray_image, Size(3, 3), 0);*/
        cv::threshold(gray_image, binary_image, 180, 255, cv::THRESH_BINARY);
        // imshow("binary_image", binary_image);
        // waitKey(0);
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        // 查找轮廓
        findContours(binary_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (vector<cv::Point> contour : contours)
        {
            // 筛选掉小轮廓
            if (contour.size() > 3 && contourArea(contour) > 20)
            {
                // 获取最小外接矩形
                cv::RotatedRect rect = cv::minAreaRect(contour);

                // 判断是否为灯条
                if (is_light(Light(rect)))
                {
                    cv::Point2f vertices[4];
                    rect.points(vertices);
                    // 绘制灯条最小外接矩形
                    /*for (int i = 0; i < 4; i++) {
                        line(roi_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
                    }*/
                    // 获取灯条的四个顶点真实坐标
                    for (int i = 0; i < 4; i++)
                    {
                        vertices[i].x += roi_tl.x;
                        vertices[i].y += roi_tl.y;
                    }
                    cv::RotatedRect rect_ = cv::RotatedRect(vertices[0], vertices[1], vertices[2]);
                    lights.push_back(Light(rect_));
                }
            }
        }
        if (lights.size() > 2)
        {
            sort(lights.begin(), lights.end(), [](const Light &a, const Light &b)
                 { return a.center.x > b.center.x; });
            // 只保留首尾灯条
            lights.erase(lights.begin() + 1, lights.end() - 1);
        }
        return lights;
    }

    // TODO:
    bool ArmorDet::is_light(const Light &light)
    {
        // 筛选掉不符合条件的灯条
        if (light.width / light.length > 0.5 || light.tilt_angle > 45)
        {
            return false;
        }
        return true;
    }

    vector<Armor> ArmorDet::match_lights(const vector<Light> &lights, vector<int> &result)
    {
        vector<Armor> armors;
        vector<string> class_names = {"red", "blue"};
        // 匹配灯条
        for (size_t i = 0; i < lights.size(); i++)
        {
            for (size_t j = i + 1; j < lights.size(); j++)
            {
                if (is_armor(lights[i], lights[j]) != ArmorType::INVALID)
                {
                    Armor armor(lights[i], lights[j]);
                    armor.classfication_result = class_names[result[4]];
                    armor.color = class_names[result[4]];
                    armor.yolo_confidence = result[5];
                    armors.push_back(armor);
                }
            }
        }
        return armors;
    }

    ArmorType ArmorDet::is_armor(const Light &light1, const Light &light2)
    {
        // 判断两根灯条是否为装甲板
        float length_ratio = light1.length < light2.length ? light1.length / light2.length : light2.length / light1.length;
        float avg_length = (light1.length + light2.length) / 2;
        float center_distance = norm(light1.center - light2.center) / avg_length;
        // TODO: 判断装甲板类型

        cv::Point2f diff = light1.center - light2.center;
        float angle = atan2(abs(diff.y), abs(diff.x)) * 180 / CV_PI;
        // cout << length_ratio << " " << center_distance << " " << angle << endl;
        bool is_armor = length_ratio > 0.4 && center_distance > 1.0 && angle < 45;
        ArmorType type;
        if (is_armor)
        {
            type = center_distance > 3.0 ? ArmorType::LARGE : ArmorType::SMALL;
        }
        else
        {
            type = ArmorType::INVALID;
        }

        return type;
    }

    PnPSolver::PnPSolver(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs)
    {
        // 初始化相机参数
        this->camera_matrix = camera_matrix;
        this->dist_coeffs = dist_coeffs;
        // 初始化物体坐标
        object_points = {cv::Point3f(0, 0, 0), cv::Point3f(0.055 / 2, -0.135 / 2, 0), cv::Point3f(-0.055 / 2, -0.135 / 2, 0),
                         cv::Point3f(-0.055 / 2, 0.135 / 2, 0), cv::Point3f(0.055 / 2, 0.135 / 2, 0)}; // m为单位
    }

    vector<double> PnPSolver::solve(const Armor &armor)
    {
        // 初始化图像坐标
        image_points = {armor.center, armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom, armor.right_light.top};
        // 解PnP
        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
        // 计算距离
        double distance = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));
        // cout << "distance: " << distance << endl;
        // 计算角度
        double yaw = atan2(tvec.at<double>(0), tvec.at<double>(2));
        double pitch = atan2(tvec.at<double>(1), tvec.at<double>(2));
        // cout << "yaw: " << yaw << " pitch: " << pitch << endl;
        vector<double> data = {armor.center.x, armor.center.y, yaw, pitch, distance};
        return data;
    }
}