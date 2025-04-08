#include <chrono>
#include <limits>

#include <armor_tracker.hpp>

extern std::atomic<bool> detect_color;

ArmorTracker::ArmorTracker(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs)
            : current_id_(0), camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs) {}

void ArmorTracker::assignAttr(cv::Mat &tvec, cv::Mat &rvec, Armor& armor) {
    armor.x = tvec.at<double>(0, 0);
    armor.y = tvec.at<double>(1, 0);
    armor.z = tvec.at<double>(2, 0);

    armor.pitch = rvec.at<double>(0, 0);
    armor.yaw = rvec.at<double>(1, 0);
    armor.roll = rvec.at<double>(2, 0);
}

int ArmorTracker::assignNewID() {
    return current_id_++;
}

void ArmorTracker::track(std::vector<Armor> &armors, cv::Mat &frame, double dt) {
    if (armors.empty()) return;
    PnPSolver pnp_solver(camera_matrix_, dist_coeffs_);
    vector<vector<double>> cost_matrix = vector<vector<double>>(armors.size(), vector<double>(tracked_armors_.size()));
    auto current_time = chrono::high_resolution_clock::now();

    for (auto& tracked_armor : tracked_armors_) {
        double dt = chrono::duration_cast<chrono::duration<double>>(current_time - tracked_armor.last_update_time).count();
        tracked_armor.ekf_tracker.predict(dt);
    }

    for (int i = 0; i < armors.size(); ++i) {
        auto armor = armors[i];
        if (armor.color != (detect_color.load() ? "red" : "blue")) continue;
        std::vector<cv::Mat> poses = pnp_solver.solve(armor);
        cv::Mat tvec = poses[0], rvec = poses[1];
        cv::drawFrameAxes(frame, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1, 1);
        assignAttr(tvec, rvec, armor);
        
        for (int j = 0; j < tracked_armors_.size(); ++j) {
            auto tracked_armor = tracked_armors_[j];
            // 获取代价矩阵
            cv::Point3d predict_position = cv::Point3d(tracked_armor.ekf_tracker.getState().at<double>(0, 0),
                                                         tracked_armor.ekf_tracker.getState().at<double>(1, 0),
                                                         tracked_armor.ekf_tracker.getState().at<double>(2, 0));
            cv::Point3d current_position = cv::Point3d(armor.x, armor.y, armor.z);
            double cost = cv::norm(predict_position - current_position);
            cost_matrix[i][j] = cost;
        }
    }

    std::vector<int> result = hungarian(cost_matrix);
    
    vector<bool> matched(tracked_armors_.size(), false);
    for (int i = 0; i < result.size(); ++i) {
        std::cout << i << "->" << result[i] << endl;
        if (result[i] != -1) {
            int j = result[i];

            double distance = cv::norm(cv::Point3f(armors[i].x, armors[i].y, armors[i].z) - cv::Point3f(tracked_armors_[j].armor.x, tracked_armors_[j].armor.y, tracked_armors_[j].armor.z));
            double tilt_angle = abs(armors[i].yaw - tracked_armors_[j].armor.yaw);
            if (distance > 0.13 || tilt_angle > 0.6) continue;

            tracked_armors_[j].ekf_tracker.update(armors[i].x, armors[i].y, armors[i].z, armors[i].yaw, armors[i].pitch);
            tracked_armors_[j].armor = armors[i];
            tracked_armors_[j].lost_count = 0;
            tracked_armors_[j].last_update_time = current_time;
            matched[j] = true;
        }
    }

    // 处理未匹配的跟踪器（标记为丢失）
    for (int j = 0; j < tracked_armors_.size(); ++j) {
        if (!matched[j]) {
            tracked_armors_[j].lost_count++;
        }
    }

    // 移除长时间丢失的跟踪器
    tracked_armors_.erase(
        remove_if(tracked_armors_.begin(), tracked_armors_.end(),
                  [](const TrackedArmor& t) { return t.lost_count > 5; }),
        tracked_armors_.end()
    );


    // 添加新检测到的装甲板作为新跟踪器
    for (auto& armor : armors) {
        if (armor.color != (detect_color.load() ? "red" : "blue")) continue;
        bool is_new = true;
        for (auto& tracked : tracked_armors_) {
            if (tracked.armor.track_id == armor.track_id) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            TrackedArmor new_tracked;
            armor.track_id = assignNewID();
            new_tracked.armor = armor;
            new_tracked.ekf_tracker.init(armor.x, armor.y, armor.z, armor.yaw, armor.pitch, dt);
            tracked_armors_.push_back(new_tracked);
        }
    }
}

void ArmorTracker::drawTrajectories(cv::Mat& frame, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
}


void ArmorTracker::dfs(int i, vector<bool>& used_j, double current_cost, double& min_cost, vector<int>& current_match, vector<int>& best_match, const vector<vector<double>>& cost_matrix) {
    if (i == cost_matrix.size()) {
        if (current_cost < min_cost) {
            min_cost = current_cost;
            best_match = current_match;
        }
        return;
    }
    for (int j = 0; j < cost_matrix[i].size(); ++j) {
        if (!used_j[j]) {
            used_j[j] = true;
            current_match[i] = j;
            dfs(i + 1, used_j, current_cost + cost_matrix[i][j], min_cost, current_match, best_match, cost_matrix);
            used_j[j] = false;
            current_match[i] = -1;
        }
    }
}

vector<int> ArmorTracker::hungarian(const vector<vector<double>>& cost_matrix) {
    int n = cost_matrix.size();
    if (n == 0) return {};
    int m = cost_matrix[0].size();

    vector<int> best_match(n, -1);
    vector<int> current_match(n, -1);
    vector<bool> used_j(m, false);
    double min_cost = numeric_limits<double>::infinity();

    dfs(0, used_j, 0.0, min_cost, current_match, best_match, cost_matrix);

    return best_match;
}