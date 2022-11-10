#pragma once

#include <iostream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace vins_slam {

using namespace std;

class Camera {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Camera(Eigen::Matrix3d &K, Eigen::Matrix4d &ext_imu_cam,double k1, double k2, double p1, double p2)
    {
        K = K;
        fx_ = K(0,0);
        cx_ = K(0,2);
        fy_ = K(1,1);
        cy_ = K(1,2);
        k1_ = k1;
        k2_ = k2;
        p1_ = p1;
        p2_ = p2;
        ext_cam_imu_ = ext_imu_cam.inverse();
    }
void project(Eigen::Vector3d &pc, Eigen::Vector2d &img_pos);
void project_undisort(Eigen::Vector3d &pc, Eigen::Vector2d &img_pos);
void projectun(Eigen::Vector3d &pc, Eigen::Vector2d img_norm_pos);
void unproject(Eigen::Vector2d &img_pos, Eigen::Vector3d &gray_dir);
void undisort(Eigen::Vector2d &img_pos, Eigen::Vector2d &undis_img_pos);
void undisort_normlize(Eigen::Vector2d &img_pos, Eigen::Vector2d &undis_img_pos);
void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u);
    Eigen::Matrix4d ext_cam_imu_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
private:
    Eigen::Matrix3d K;
    double k1_;
    double k2_;
    double p1_;
    double p2_;

};

}