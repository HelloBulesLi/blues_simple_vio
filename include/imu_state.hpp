#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace vins_slam {

struct IMU_State {
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    unsigned long int id;
    double timestamp;

    // rotation from world frame to body(imu) frame
    Eigen::Quaterniond rotation_q;
    Eigen::Vector4d rotation_vec;
    Eigen::Matrix3d rotation_matrix;
    // body(imu) frame position in world frame
    Eigen::Vector3d position;

    // body(imu) frame velocity in world frame
    Eigen::Vector3d velocity;

    // angle vel bias
    Eigen::Vector3d bg;

    // acc vel bias
    Eigen::Vector3d ba;

    // rotation null space
    Eigen::Vector4d rotation_null;
    Eigen::Vector3d position_null;
    Eigen::Vector3d velocity_null;

    // rotation from body(imu) frame to camera frame
    Eigen::Matrix3d R_cam0_imu;
    Eigen::Vector3d t_imu_cam0;

    // IMU state construct
    IMU_State(): id(0), timestamp(0),
        rotation_vec(Eigen::Vector4d(0,0,0,1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        bg(Eigen::Vector3d::Zero()),
        ba(Eigen::Vector3d::Zero()),
        rotation_null(Eigen::Vector4d(0,0,0,1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}
    
    IMU_State(const unsigned long int &new_id): id(new_id), timestamp(0),
        rotation_vec(Eigen::Vector4d(0,0,0,1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        bg(Eigen::Vector3d::Zero()),
        ba(Eigen::Vector3d::Zero()),
        rotation_null(Eigen::Vector4d(0,0,0,1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}

};

}
