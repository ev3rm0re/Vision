#include <boost/math/tools/roots.hpp>
#include "predictor.hpp"

void Predictor::getAttr(std::vector<cv::Mat> &vec, Armor& armor) {
    armor.x = vec.at(0).at<double>(0, 0);
    armor.y = vec.at(0).at<double>(1, 0);
    armor.z = vec.at(0).at<double>(2, 0);

    armor.yaw = vec.at(1).at<double>(0, 0);
    armor.pitch = vec.at(1).at<double>(1, 0);
    armor.roll = vec.at(1).at<double>(2, 0);
}

bool Predictor::calculate(Armor& last_armor, Armor& armor, double interval) {
    alpha0 = atan2(last_armor.x, last_armor.z);
    alpha1 = atan2(armor.x, armor.z);
    double delta_alpha = alpha1 - alpha0;

    beta0 = last_armor.yaw;
    beta1 = armor.yaw;

    x0 = last_armor.x;
    x1 = armor.x;
    z0 = last_armor.z;
    z1 = armor.z;

    double delta_theta = beta1 - beta0;
    omega = delta_theta / interval;

    d0 = sqrt(last_armor.x * last_armor.x + last_armor.z * last_armor.z);
    d1 = sqrt(armor.x * armor.x + armor.z * armor.z);

    r = sqrt((d0 * d0 + d1 * d1 - 2 * d0 * d1 * cos(delta_alpha)) / (2 - 2 * cos(delta_theta)));
    dc = sqrt(r * r + d1 * d1 - 2 * r * d1 * cos(CV_PI - beta1 + alpha1));
    
    double zc = armor.z * (armor.z / cos(beta1) + r) / (armor.z / cos(beta1));

    if (abs(dc - d1) / d1 > 0.08) return false;
    std::cout << "delta_alpha: " << delta_alpha << " delta_theta: " << delta_theta << " omega: " << omega << std::endl;
    std::cout << "last_armor.x: " << last_armor.x << " last_armor.z: " << last_armor.z << " d0: " << d0 << std::endl;
    std::cout << "armor.x: " << armor.x << " armor.z: " << armor.z << " d1: " << d1 << std::endl;
    std::cout << "r: " << r << " dc: " << dc << std::endl;
    std::cout << "armor current yaw: " << alpha1 + CV_PI / 4.0 << std::endl;
    return true;
}

bool Predictor::predict(double& aim_yaw, double& aim_pitch) {
    Equation f(x0, x1, z0, z1, r, v, beta0, beta1, omega);
    double t_lower = 1e-2;
    double t_upper = 1.0;

    boost::uintmax_t max_iter = 100;
    try {
        auto result = boost::math::tools::bisect(f, t_lower, t_upper, boost::math::tools::eps_tolerance<double>(5), max_iter);
        t_solution = (result.first + result.second) / 2.0;
        std::cout << "t_solution: " << t_solution << std::endl;

        double d2 = v * t_solution;
        double xt = ((x1 + r * sin(beta1) - r * sin(CV_PI / 2 - beta1 - omega * t_solution)) + (x0 + r * sin(beta0) - r * sin(CV_PI / 2 - beta1 - omega * t_solution))) / 2.0;
        double zt = ((z1 + r * cos(beta1) - r * cos(CV_PI / 2 - beta1 - omega * t_solution)) + (z0 + r * cos(beta0) - r * cos(CV_PI / 2 - beta1 - omega * t_solution))) / 2.0;
        aim_yaw = atan2(xt, zt);
        std::cout << "aim_yaw: " << aim_yaw << std::endl; // 相机下一次击打点的yaw
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    
    return true;
}

void Predictor::drawPrediction(const cv::Mat &camera_matrix, cv::Mat &frame) {
    double xt = ((x1 + r * sin(beta1) - r * sin(CV_PI / 2 - beta1 - omega * t_solution)) + (x0 + r * sin(beta0) - r * sin(CV_PI / 2 - beta1 - omega * t_solution))) / 2.0;
    double zt = ((z1 + r * cos(beta1) - r * cos(CV_PI / 2 - beta1 - omega * t_solution)) + (z0 + r * cos(beta0) - r * cos(CV_PI / 2 - beta1 - omega * t_solution))) / 2.0;
    double u = camera_matrix.at<double>(0, 0) * xt / zt + camera_matrix.at<double>(0, 2);
    std::cout << "u: " << u << std::endl;
    cv::circle(frame, cv::Point(u, camera_matrix.at<double>(1, 2)), 10, cv::Scalar(0, 0, 255), -1);
}