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

    x = armor.x;
    z = armor.z;

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
    return true;
}

bool Predictor::predict(double& aim_yaw, double& aim_pitch) {
    Equation f(x, z, r, d1, v, alpha1, beta1, omega);
    double t_lower = 1e-2;
    double t_upper = 1.0;

    boost::uintmax_t max_iter = 500;
    try {
        auto result = boost::math::tools::bisect(f, t_lower, t_upper, boost::math::tools::eps_tolerance<double>(5), max_iter);
        double t_solution = (result.first + result.second) / 2.0;
        std::cout << "t_solution: " << t_solution << std::endl;

        double d2 = v * t_solution;
        double temp = (d2 * d2 - d1 * d1 - 2 * d1 * r * cos(CV_PI - beta1 + alpha1)) / (2 * r * d2);
        if (temp > 1 || temp < -1) return false;
        aim_yaw = acos(temp) - beta1 - omega * t_solution;
        std::cout << "aim_yaw: " << aim_yaw << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    
    return true;
}