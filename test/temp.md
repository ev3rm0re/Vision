```cpp
        //         double yaw = atan2(armor.x, armor.z);
        //         double pitch = atan2(armor.y, armor.z);
        //         // 初始化或更新 EKF Tracker
        //         if (ekf_trackers.find(armor.track_id) == ekf_trackers.end()) {
        //             ekf_trackers[armor.track_id].init(armor.x, armor.y, armor.z, yaw, pitch, 1.0 / 80.0);
        //         } else {
        //             ekf_trackers[armor.track_id].predict(1.0 / 80.0);
        //             ekf_trackers[armor.track_id].update(armor.x, armor.y, armor.z, yaw, pitch);
        //         }
        //         cv::Mat ekf_state = ekf_trackers[armor.track_id].getState();
        //         double filtered_x = ekf_state.at<double>(0, 0);
        //         double filtered_y = ekf_state.at<double>(1, 0);
        //         double filtered_z = ekf_state.at<double>(2, 0);
        //         double filtered_yaw = ekf_state.at<double>(6, 0);
        //         double filtered_pitch = ekf_state.at<double>(8, 0);
        //         cv::Point3f current_point(filtered_x, filtered_y, filtered_z);
        //         // 如果轨迹非空，则获取最后一个轨迹点
        //         if (!id_trajectory[armor.track_id].empty()) {
        //             cv::Point3f last_traj_point = id_trajectory[armor.track_id].back();
        //             double distance = cv::norm(last_traj_point - current_point);
        //             double tilt_angle = abs(last_armor.yaw - armor.yaw);
        //             if (distance > 0.13 || tilt_angle > 0.6 || lost_count > 4) {
        //                 int old_id = armor.track_id;
        //                 int new_id = ++current_id;
        //                 armor.track_id = new_id;
        //                 // 删除旧的轨迹并初始化新轨迹
        //                 id_trajectory.erase(old_id);
        //                 id_trajectory[new_id] = std::vector<cv::Point3f>();
        //                 // 更新 tracked_armors 中的记录
        //                 tracked_armors.erase(old_id);
        //                 tracked_armors[new_id] = armor;
        //                 // cout << ">>>>>>>Armor ID updated from " << old_id << " to " << new_id << endl;
        //             } else {
        //                 // 无跳变时更新 tracked 信息
        //                 tracked_armors[armor.track_id] = armor;
        //             }
        //         } else {
        //             // 如果轨迹为空，直接更新 tracked 信息
        //             tracked_armors[armor.track_id] = armor;
        //         }
        //         // 将当前 EKF 预测点添加到对应ID的轨迹中
        //         if (id_trajectory.find(armor.track_id) == id_trajectory.end()) {
        //             id_trajectory[armor.track_id] = std::vector<cv::Point3f>();
        //         }
        //         id_trajectory[armor.track_id].push_back(current_point);
        //         if (id_trajectory[armor.track_id].size() > 500) {
        //             id_trajectory[armor.track_id].erase(id_trajectory[armor.track_id].begin());
        //         }
        //         // // 绘制当前目标轨迹（只绘制当前ID的轨迹）
        //         // auto& trajectory = id_trajectory[armor.track_id];
        //         // for (size_t i = 1; i < trajectory.size(); i++) {
        //         //     cv::line(frame, trajectory[i - 1], trajectory[i], cv::Scalar(0, 255, 0), 2);
        //         // }
        //         // // 绘制当前位置、箭头及其他信息
        //         // cv::circle(frame, cv::Point(filtered_x, filtered_y), 5, cv::Scalar(0, 0, 255), -1);
        //         // double vx = ekf_state.at<double>(3, 0);
        //         // double vy = ekf_state.at<double>(4, 0);
        //         // double angle = atan2(vy, vx);
        //         // cv::Point2f arrow_end(filtered_x + 30 * cos(angle), filtered_y + 30 * sin(angle));
        //         // cv::arrowedLine(frame, cv::Point(filtered_x, filtered_y), arrow_end, cv::Scalar(255, 255, 255), 2);
        //         // cv::putText(frame, "Tracking Target", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        //         if (last_armor.track_id == armor.track_id) {
        //             double aim_yaw, aim_pitch;
        //             bool success = predictor.calculate(last_armor, armor, duration);
        //             if (success) {
        //                 predictor.predict(aim_yaw, aim_pitch);
        //                 predictor.drawPrediction(camera_matrix, frame);
        //                 if (serial_ready) {
        //                     send_packet.yaw = aim_yaw;
        //                     send_packet.pitch = aim_pitch;                 
        //                     send_packet.tracking = false;
        //                     send_packet.id = id_unit8_map.at(armor.number);
        //                     send_packet.armors_num = 4;
        //                     send_packet.reserved = 0;
        //                     serial_thread.send_packet(send_packet);
        //                 }
        //             }
        //         }
        //         line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 1);
        //         line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 1);
        //         putText(frame, "armor id: " + to_string(armor.track_id), armor.right_light.top + cv::Point2f(5, -40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //         putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //         putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //         putText(frame, "distance: " + to_string(armor.z).substr(0, 3), armor.right_light.center + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //         last_armor = armor;
        //         last_timestamp = timestamp;
        //     }
        // } else {
        //     lost_count++;
        // }


    for (auto& armor : armors) {
        auto armor = tracedArmor.second;
        line(frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 1);
        line(frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 1);
        putText(frame, "armor id: " + to_string(armor.id), armor.right_light.top + cv::Point2f(5, -40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        putText(frame, armor.classfication_result, armor.right_light.top + cv::Point2f(5, -20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        putText(frame, "yolo conf: " + to_string(armor.yolo_confidence).substr(0, 2) + "%", armor.right_light.center + cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        putText(frame, "distance: " + to_string(armor.z).substr(0, 3), armor.right_light.center + cv::Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        last_armor = armor;
        last_timestamp = timestamp;
    }

```