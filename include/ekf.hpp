#ifndef _EKF_HPP_
#define _EKF_HPP_

#include <opencv2/opencv.hpp>

class EKFTracker {
    public:
        // 初始化状态向量，假设初始速度和角速度为0
        void init(double x, double y, double z, double yaw, double pitch, double dt) {
            // 状态向量维度：[x, y, z, vx, vy, vz, yaw, yaw_rate, pitch, pitch_rate]
            state = cv::Mat::zeros(10, 1, CV_64F);
            state.at<double>(0) = x;         // x
            state.at<double>(1) = y;         // y
            state.at<double>(2) = z;         // z
            state.at<double>(6) = yaw;       // yaw
            state.at<double>(8) = pitch;     // pitch
    
            // 状态协方差矩阵 P（根据需要初始化）
            P = cv::Mat::eye(10, 10, CV_64F) * 0.1;
    
            // 状态转移矩阵 F（10x10）
            F = cv::Mat::eye(10, 10, CV_64F);
    
            // 测量矩阵 H：假设测量为 [x, y, z, yaw, pitch]
            H = cv::Mat::zeros(5, 10, CV_64F);
            H.at<double>(0, 0) = 1.0;  // 测量 x
            H.at<double>(1, 1) = 1.0;  // 测量 y
            H.at<double>(2, 2) = 1.0;  // 测量 z
            H.at<double>(3, 6) = 1.0;  // 测量 yaw
            H.at<double>(4, 8) = 1.0;  // 测量 pitch
    
            // 观测噪声 R（5x5）
            R = cv::Mat::eye(5, 5, CV_64F) * 0.05;
    
            // 过程噪声 Q（10x10），根据系统特性设置
            Q = cv::Mat::eye(10, 10, CV_64F) * 0.01;
    
            // 更新状态转移矩阵 F 中的 dt
            F.at<double>(0, 3) = dt;  // 位置预测 x -> vx
            F.at<double>(1, 4) = dt;  // 位置预测 y -> vy
            F.at<double>(2, 5) = dt;  // 位置预测 z -> vz
            F.at<double>(3, 3) = 1.0; // 速度 vx 保持为1
            F.at<double>(4, 4) = 1.0; // 速度 vy 保持为1
            F.at<double>(5, 5) = 1.0; // 速度 vz 保持为1
            F.at<double>(6, 7) = dt;  // yaw -> yaw_rate
            F.at<double>(7, 7) = 1.0; // yaw_rate 保持为1
            F.at<double>(8, 9) = dt;  // pitch -> pitch_rate
            F.at<double>(9, 9) = 1.0; // pitch_rate 保持为1
        }
    
        // 预测步骤，根据时间间隔 dt 更新状态
        void predict(double dt) {
            // 更新状态转移矩阵 F 中的时间间隔（如果 dt 动态变化）
            F.at<double>(0, 3) = dt;
            F.at<double>(1, 4) = dt;
            F.at<double>(2, 5) = dt;
            F.at<double>(6, 7) = dt;
            F.at<double>(8, 9) = dt;
    
            // 状态预测: x = F * x
            state = F * state;
    
            // 协方差预测: P = F * P * F' + Q
            P = F * P * F.t() + Q;
        }
    
        // 更新步骤，传入测量值 [x, y, z, yaw, pitch]
        void update(double x, double y, double z, double yaw, double pitch) {
            cv::Mat z_meas = (cv::Mat_<double>(5, 1) << x, y, z, yaw, pitch);
            // 计算残差: y = z - H * x
            cv::Mat y_residual = z_meas - H * state;
            // 计算卡尔曼增益: K = P * H' * (H * P * H' + R)^-1
            cv::Mat S = H * P * H.t() + R;
            cv::Mat K = P * H.t() * S.inv();
            // 更新状态: x = x + K * y
            state = state + K * y_residual;
            // 更新协方差: P = (I - K * H) * P
            cv::Mat I = cv::Mat::eye(P.size(), P.type());
            P = (I - K * H) * P;
        }
    
        cv::Mat getState() const { return state; }
    
    private:
        cv::Mat state; // 状态向量 [x, y, z, vx, vy, vz, yaw, yaw_rate, pitch, pitch_rate]
        cv::Mat P;     // 协方差矩阵
        cv::Mat F;     // 状态转移矩阵
        cv::Mat H;     // 测量矩阵
        cv::Mat R;     // 观测噪声矩阵
        cv::Mat Q;     // 过程噪声矩阵
};


#endif // _EKF_HPP_
