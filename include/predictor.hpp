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
        double v = 23.0;
        double omega;
        double x;
        double z;
    private:
        double alpha0;
        double alpha1;
        double beta0;
        double beta1;
        double d0;
        double d1;
        double r;
        double dc;
};

// 定义一个结构体，用于封装方程 f(t)=0
struct Equation {
    double x, z, r, d1, v, alpha1, beta1, omega;

    Equation(double x_, double z_, double r_, double d_, double v_, double alpha_, double beta_, double omega_)
        : x(x_), z(z_), r(r_), d1(d_), v(v_), alpha1(alpha_), beta1(beta_), omega(omega_) {}

    // 定义函数运算符，计算 f(t)
    double operator()(double t) const {
        double lhs = acos((v * v * t * t - d1 * d1 + 2 * d1 * r * cos(CV_PI - beta1 + alpha1)) / (2 * r * v * t)) - beta1 - omega * t;

        double rhs = atan2((x + r * sin(beta1) + r * cos(omega * t - CV_PI / 2 + beta1)), (z + r * cos(beta1) - r * sin(omega * t - CV_PI / 2 + beta1)));
        return lhs - rhs;
    }
};

#endif
