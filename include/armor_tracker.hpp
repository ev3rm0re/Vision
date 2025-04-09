#ifndef _TRACKER_HPP_
#define _TRACKER_HPP_

#include <opencv2/opencv.hpp>

#include <armor.hpp>
#include <packet.hpp>
#include <ekf.hpp>
#include <detector.hpp>

class ArmorTracker{
public:
    struct TrackedArmor {
        Armor armor;
        EKFTracker ekf_tracker;
        int lost_count;
        chrono::time_point<chrono::high_resolution_clock> last_update_time;
    };
    ArmorTracker(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs);
    void assignAttr(cv::Mat &tvec, cv::Mat &rvec, Armor& armor);
    void track(std::vector<Armor> &armors, cv::Mat &frame);
    vector<int> hungarian(const vector<vector<double>>& cost_matrix);

private:
    int assignNewID();

    std::vector<TrackedArmor> tracked_armors_;
    int current_id_;
    const cv::Mat &camera_matrix_;
    const cv::Mat &dist_coeffs_;
};

#endif