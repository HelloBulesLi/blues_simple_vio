#pragma once

#include "eigen3/Eigen/Dense"

namespace vins_slam {
class IMU_Measure {
public:
    IMU_Measure(Eigen::Vector3d &ang_vel, Eigen::Vector3d &acc_vel, double timestamp);
    // ~IMU_Measure() {};
    Eigen::Vector3d angular_vel;
    Eigen::Vector3d accelarator;
    double timestamp_;
    int imu_measure_id_;
};

}