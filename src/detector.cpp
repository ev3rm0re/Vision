#include "detector.hpp"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;
using namespace ov;

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

    Mat YoloDet::letterbox(const Mat &source)
    {
        // 将图像填充为正方形
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        Mat result = Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(Rect(0, 0, col, row)));
        return result;
    }

    Tensor YoloDet::infer(const Mat &image)
    {
        // 推理

        Mat letterbox_image = YoloDet::letterbox(image);
        scale = letterbox_image.size[0] / 640.0;
        Mat blob = dnn::blobFromImage(letterbox_image, 1.0 / 255.0, Size(640, 640), Scalar(), true);
        auto &input_port = compiled_model.input();
        Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto output = infer_request.get_output_tensor(0);
        return output;
    }

    vector<vector<int>> YoloDet::postprocess(const Tensor &output, const float &score_threshold, const float &iou_threshold) const
    {
        // 后处理
        float *data = output.data<float>();
        Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, data);
        transpose(output_buffer, output_buffer);
        vector<int> class_ids;
        vector<float> class_scores;
        vector<Rect> boxes;
        vector<vector<int>> results;
        // 遍历输出层
        for (int i = 0; i < output_buffer.rows; i++)
        {
            // 获取类别得分
            Mat classes_scores = output_buffer.row(i).colRange(4, 6);
            Point class_id;
            double maxClassScore;
            // 获取最大类别得分和类别索引
            minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > score_threshold)
            {
                // 将类别得分和类别索引存储
                class_scores.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                // 获取边界框
                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);
                // 计算边界框真实坐标
                int left = int((cx - 0.5 * w) * scale);
                int top = int((cy - 0.5 * h) * scale);
                int width = int(w * scale);
                int height = int(h * scale);
                // 将边界框存储
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        vector<int> indices;
        // 非极大值抑制
        dnn::NMSBoxes(boxes, class_scores, score_threshold, iou_threshold, indices);
        for (size_t i = 0; i < indices.size(); i++)
        {
            results.push_back(vector<int>{boxes[indices[i]].tl().x, boxes[indices[i]].tl().y, boxes[indices[i]].br().x, boxes[indices[i]].br().y, class_ids[indices[i]], (int)(class_scores[indices[i]] * 100)});
        }
        return results;
    }

    vector<Armor> ArmorDet::detect(const vector<vector<int>> &results, const Mat &image)
    {
        vector<Armor> armors;
        vector<Armor> armors_;

        // 遍历检测结果
        for (vector<int> result : results)
        {
            // 获取检测结果的ROI
            Rect roi = Rect(result[0], result[1], result[2] - result[0], result[3] - result[1]);
            // 筛选掉超出图像范围的ROI
            if (roi.x < 0 || roi.x + roi.width > image.cols || roi.y < 0 || roi.y + roi.height > image.rows)
            {
                continue;
            }
            // 获取ROI图像
            Mat roi_image = image(roi);
            // 获取ROI左上角坐标
            Point2f roi_tl = Point2f(result[0], result[1]);
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

    vector<Light> ArmorDet::find_lights(const Mat &roi_image, const Point2f &roi_tl)
    {
        vector<Light> lights;
        Mat gray_image, binary_image;
        // 灰度化、高斯模糊、二值化
        cvtColor(roi_image, gray_image, COLOR_BGR2GRAY);
        /*GaussianBlur(gray_image, gray_image, Size(3, 3), 0);*/
        threshold(gray_image, binary_image, 180, 255, THRESH_BINARY);
        // imshow("binary_image", binary_image);
        // waitKey(0);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        // 查找轮廓
        findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (vector<Point> contour : contours)
        {
            // 筛选掉小轮廓
            if (contour.size() > 3 && contourArea(contour) > 20)
            {
                // 获取最小外接矩形
                RotatedRect rect = minAreaRect(contour);

                // 判断是否为灯条
                if (is_light(Light(rect)))
                {
                    Point2f vertices[4];
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
                    RotatedRect rect_ = RotatedRect(vertices[0], vertices[1], vertices[2]);
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

        Point2f diff = light1.center - light2.center;
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

    PnPSolver::PnPSolver(const Mat &camera_matrix, const Mat &dist_coeffs)
    {
        // 初始化相机参数
        this->camera_matrix = camera_matrix;
        this->dist_coeffs = dist_coeffs;
        // 初始化物体坐标
        object_points = {Point3f(0, 0, 0), Point3f(55 / 2, -135 / 2, 0), Point3f(-55 / 2, -135 / 2, 0),
                         Point3f(-55 / 2, 135 / 2, 0), Point3f(55 / 2, 135 / 2, 0)}; // mm为单位
    }

    vector<double> PnPSolver::solve(const Armor &armor)
    {
        // 初始化图像坐标
        image_points = {armor.center, armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom, armor.right_light.top};
        // 解PnP
        solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
        // 计算距离
        double distance = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));
        cout << "distance: " << distance << endl;
        // 计算角度
        double yaw = atan2(tvec.at<double>(0), tvec.at<double>(2));
        double pitch = atan2(tvec.at<double>(1), tvec.at<double>(2));
        cout << "yaw: " << yaw << " pitch: " << pitch << endl;
        vector<double> data = {yaw, pitch, distance};
        return data;
    }
}