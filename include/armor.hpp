/*
装甲板结构体定义
1. Light: 灯条结构体
2. Armor: 装甲板结构体
*/

#ifndef _ARMOR_HPP_
#define _ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

enum class ArmorType {
    SMALL,
    LARGE,
    INVALID
};

struct Light : public cv::RotatedRect {
    Light() = default;
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box) {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                  { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);
        area = length * width;
        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color = 0;
    cv::Point2f top, bottom;
    double length = 0.0;
    double width = 0.0;
    double area = 0.0;
    float tilt_angle = 0.0;
};

struct Armor {
    Armor() = default;
    Armor(const Light &l1, const Light &l2) {
        if (l1.center.x < l2.center.x) {
            left_light = l1, right_light = l2;
        } else {
            left_light = l2, right_light = l1;
        }
        center = (left_light.center + right_light.center) / 2;
    }

    // 灯条
    Light left_light, right_light;
    cv::Point2f center;

    ArmorType type = ArmorType::SMALL;

    // 数字识别
    cv::Mat number_img;
    std::string number;
    float confidence = 0.0;
    float yolo_confidence = 0.0;
    std::string classfication_result;
    std::string color;
    
    int id = -1;

    // 装甲板自身相对于相机坐标系的位姿
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

#endif