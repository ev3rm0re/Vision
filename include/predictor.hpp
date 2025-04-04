#ifndef _CALCULATOR_HPP_
#define _CALCULATOR_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

#include <armor.hpp>

class Predictor {
    public:
        Predictor() = default;
        void getAttr(std::vector<cv::Mat> &vec, Armor& armor);
        bool calculate(Armor& last_armor, Armor& armor, double interval);
        bool predict(double& aim_yaw, double& aim_pitch);
        void drawPrediction(const cv::Mat &camera_matrix, cv::Mat &frame);
        double v = 23.0;
        double omega;
        double x0;
        double z0;
        double x1;
        double z1;
    private:
        double alpha0;
        double alpha1;
        double beta0;
        double beta1;
        double d0;
        double d1;
        double r;
        double dc;
        double t_solution;
};

// 定义一个结构体，用于封装方程 f(t)=0
struct Equation {
    double x0, x1, z0, z1, r, v, beta0, beta1, omega;

    Equation(double x0, double x1, double z0, double z1, double r, double v, double beta0, double beta1, double omega)
        : x0(x0), x1(x1), z0(z0), z1(z1), r(r), v(v), beta0(beta0), beta1(beta1), omega(omega) {}
    // 定义函数运算符，计算 f(t)
    double operator()(double t) const {
        double lhs = v * v * t * t;

        double xt = ((x1 + r * sin(beta1) - r * sin(CV_PI / 2 - beta1 - omega * t)) + (x0 + r * sin(beta0) - r * sin(CV_PI / 2 - beta1 - omega * t))) / 2.0;
        double zt = ((z1 + r * cos(beta1) - r * cos(CV_PI / 2 - beta1 - omega * t)) + (z0 + r * cos(beta0) - r * cos(CV_PI / 2 - beta1 - omega * t))) / 2.0;

        double rhs = xt * xt + zt * zt;
        return lhs - rhs;
    }
};

#endif
