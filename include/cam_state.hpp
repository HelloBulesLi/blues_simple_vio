#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <map>

namespace vins_slam {

struct Cam_State {
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned long int id;

    double timestamp;
    
    // rotation from world frame to camera frame
    Eigen::Vector4d rotation_vec;
    Eigen::Quaterniond rotation_q;
    Eigen::Matrix3d rotation_matrix;

    // position of camera frame in world frame
    Eigen::Vector3d position;

    // null space of postion and rotate
    Eigen::Vector4d rotation_null;
    Eigen::Vector3d position_null;

    // Cam state construct
    Cam_State(): id(0), timestamp(0),
        rotation_vec(Eigen::Vector4d(0,0,0,1)),
        position(Eigen::Vector3d::Zero()),
        rotation_null(Eigen::Vector4d::Zero()),
        position_null(Eigen::Vector3d::Zero()) {};

    Cam_State(unsigned long int new_id): id(new_id), timestamp(0),
        rotation_vec(Eigen::Vector4d(0,0,0,1)),
        position(Eigen::Vector3d::Zero()),
        rotation_null(Eigen::Vector4d::Zero()),
        position_null(Eigen::Vector3d::Zero()) {};
};

typedef std::map<unsigned long int, Cam_State, std::less<unsigned long int>,
                Eigen::aligned_allocator<std::pair<unsigned long int, Cam_State>>> CamStateSever;

}