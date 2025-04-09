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

void ArmorTracker::track(vector<Armor>& armors, cv::Mat& frame) {
    auto current_time = chrono::high_resolution_clock::now();
    
    // 预测现有跟踪器
    for (auto& tracker : tracked_armors_) {
        double dt = chrono::duration_cast<chrono::duration<double>>(current_time - tracker.last_update_time).count();
        tracker.ekf_tracker.predict(dt);
    }

    // 处理检测结果
    vector<vector<double>> cost_matrix;
    PnPSolver pnp_solver(camera_matrix_, dist_coeffs_);

    if (!armors.empty()) {
        cost_matrix.resize(armors.size(), vector<double>(tracked_armors_.size(), INFINITY));
        
        for (size_t i = 0; i < armors.size(); ++i) {
            Armor& armor = armors[i];
            if (armor.color != (detect_color ? "red" : "blue")) continue;

            // 求解目标位姿
            auto poses = pnp_solver.solve(armor);
            assignAttr(poses[0], poses[1], armor);
            drawFrameAxes(frame, camera_matrix_, dist_coeffs_, poses[1], poses[0], 0.1, 2);
            cv::putText(frame, to_string(armor.yaw), armor.right_light.center + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            // 构建代价矩阵
            for (size_t j = 0; j < tracked_armors_.size(); ++j) {
                auto& tracker = tracked_armors_[j];
                cv::Point3d predict(tracker.ekf_tracker.getState().at<double>(0),
                                tracker.ekf_tracker.getState().at<double>(1),
                                tracker.ekf_tracker.getState().at<double>(2));
                
                double distance = cv::norm(predict - cv::Point3d(armor.x, armor.y, armor.z));
                double angle_diff = abs(armor.yaw - tracker.ekf_tracker.getState().at<double>(6));

                // 综合距离和角度差异，设置匹配阈值
                if (distance < 0.3 && angle_diff < 0.6) {
                    cost_matrix[i][j] = distance * 0.2 + angle_diff * 0.8;
                } else {
                    cost_matrix[i][j] = numeric_limits<double>::max();
                }
            }
        }
    }
    
    // 数据关联
    vector<int> match_result = hungarian(cost_matrix);
    vector<bool> matched(tracked_armors_.size(), false);

    // 更新匹配到的跟踪器
    for (size_t i = 0; i < match_result.size(); ++i) {
        int j = match_result[i];
        if (j != -1 && cost_matrix[i][j] < INFINITY) {
            tracked_armors_[j].ekf_tracker.update(armors[i].x, armors[i].y, armors[i].z, 
                                                armors[i].yaw, armors[i].pitch);
            armors[i].track_id = tracked_armors_[j].armor.track_id;
            tracked_armors_[j].armor = armors[i];
            tracked_armors_[j].lost_count = 0;
            tracked_armors_[j].last_update_time = current_time;
            matched[j] = true;
            armors[i].track_id = tracked_armors_[j].armor.track_id;
        } else {
            // 初始化新跟踪器
            armors[i].track_id = assignNewID();
            TrackedArmor new_tracked_armor;
            new_tracked_armor.armor = armors[i];
            new_tracked_armor.lost_count = 0;
            new_tracked_armor.last_update_time = current_time;
            new_tracked_armor.ekf_tracker = EKFTracker();
            new_tracked_armor.ekf_tracker.init(armors[i].x, armors[i].y, armors[i].z, armors[i].yaw, armors[i].pitch, 1.0 / 70.0);
            tracked_armors_.emplace_back(new_tracked_armor);
        }
    }

    for (int i = 0; i < matched.size(); i++) {
        if (matched[i] == false) {
            tracked_armors_[i].lost_count++;
        }
    }

    for (int i = 0; i < tracked_armors_.size(); i++) {
        if (tracked_armors_[i].lost_count > 1) {
            tracked_armors_.erase(tracked_armors_.begin() + i);
            i--;
        }
    }
}

vector<int> ArmorTracker::hungarian(const vector<vector<double>>& cost_matrix) {
    const double INF = numeric_limits<double>::max();
    const double EPS = 1e-9; // 浮点比较容差

    // 边界条件处理
    if (cost_matrix.empty()) return {};
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();
    if (m == 0) return vector<int>(n, -1);

    // 有效性检查：标记可匹配行
    vector<bool> valid_rows(n, false);
    for (int i = 0; i < n; ++i) {
        // 检查是否存在有效匹配（至少有一个非INF值）
        if (any_of(cost_matrix[i].begin(), cost_matrix[i].end(),
                   [&](double v) { return v < INF - EPS; })) {
            valid_rows[i] = true;
        }
    }

    // 构建带保护垫的矩阵 (n+1)x(m+1)
    vector<vector<double>> a(n+1, vector<double>(m+1, INF));
    for (int i = 1; i <= n; ++i) {
        if (!valid_rows[i-1]) continue; // 跳过无效行
        for (int j = 1; j <= m; ++j) {
            a[i][j] = cost_matrix[i-1][j-1];
        }
    }

    // KM算法核心实现
    vector<double> u(n+1, 0), v(m+1, 0);
    vector<int> p(m+1, 0), way(m+1, 0);
    
    for (int i = 1; i <= n; ++i) {
        if (!valid_rows[i-1]) { // 跳过不可匹配行
            continue;
        }

        p[0] = i;
        int j0 = 0;
        vector<double> minv(m+1, INF);
        vector<bool> used(m+1, false);

        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = 0;

            // 阶段1：寻找最小有效边
            for (int j = 1; j <= m; ++j) {
                if (!used[j] && a[i0][j] < INF - EPS) { // 忽略INF边
                    double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j] - EPS) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta - EPS) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // 关键修复：检测无效路径
            if (delta >= INF - EPS) {
                p[0] = 0;    // 标记当前行匹配失败
                break;       // 退出当前行处理
            }

            // 阶段2：顶标更新
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // 阶段3：路径回溯
        if (p[0] != 0) { // 仅处理成功匹配的行
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
    }

    // 构建结果
    vector<int> result(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0 && p[j] <= n) {
            result[p[j]-1] = j-1;
        }
    }
    return result;
}