/*
装甲板检测到之后的数字分类器
*/

#ifndef _NUMBER_CLASSIFIER_HPP_
#define _NUMBER_CLASSIFIER_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <armor.hpp>

class NumberClassifier {
public:
    NumberClassifier(
        const std::string &model_path, const std::string &label_path, const double threshold,
        const std::vector<std::string> &ignore_classes = {});
    void extractNumbers(const cv::Mat &src, std::vector<Armor> &armors);
    void classify(std::vector<Armor> &armors);

private:
    double threshold;
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    std::vector<std::string> ignore_classes_;
};

#endif