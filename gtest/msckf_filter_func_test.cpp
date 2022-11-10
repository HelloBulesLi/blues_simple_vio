#include <gtest/gtest.h>
#include <glog/logging.h>
#include "msckf_filter.hpp"
#include <eigen3/Eigen/Core>
#include <fstream>
#include "math_utils.hpp"

using namespace vins_slam;
using namespace std;
using namespace testing;
using namespace Eigen;

void processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc, std::shared_ptr<OC_MSCKF_Filter> filter_ptr);

// TODO:add a test for init mappoint, add a imu progate test
int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);

    GTEST_FLAG(filter) = "jacobain_calc_test.simulate_test";

    FLAGS_log_dir = "/home/libo/vins_msckf/build";
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging("jacobain_calc_func_test");
    RUN_ALL_TESTS();
}

// test the measurement jacobain use the simulate data
TEST(jacobain_calc_test, simulate_test)
{
    std::shared_ptr<OC_MSCKF_Filter> filter_ptr(new OC_MSCKF_Filter());

    // generate a cam_states
    Cam_State test_cam_state;
    
    Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twc = Eigen::Vector3d(1.0,1.0,1.0);
    test_cam_state.rotation_matrix = Rcw;
    test_cam_state.position = twc;

    Eigen::Matrix4d T_lcam_w = Eigen::Matrix4d::Identity();
    T_lcam_w.block<3,3>(0,0) = Rcw;
    T_lcam_w.block<3,1>(0,3) = -Rcw*twc; 

    // generate a world frame position
    Eigen::Vector3d pw(2.0, 5.0, 2.0);

    Eigen::Vector3d lPc = Rcw*pw - Rcw*twc;

    double lu = lPc(0)/lPc(2);
    double lv = lPc(1)/lPc(2);

    Eigen::Vector3d rPc = filter_ptr->R_rcam_lcam*lPc + filter_ptr->t_rcam_lcam;

    double ru = rPc(0)/rPc(2);
    double rv = rPc(1)/rPc(2);

    obs_data_t obs;
    obs.un_u0 = lu;
    obs.un_v0 = lv;
    obs.un_u1 = ru;
    obs.un_v1 = rv;

    // add a disturbin to world frame position or camere pose in world frame, 
    // get the observe by jacobain and camera project matrix,
    // compare the final result of measurement
    Eigen::Matrix<double,4,6> Hxi;
    Eigen::Matrix<double,4,3> Hj;
    Eigen::Vector4d rj;

    filter_ptr->MeasureJacobian(test_cam_state, obs, pw, Hxi, Hj, rj);

    Eigen::Matrix<double,6,1> delta_pose;
    delta_pose << 0,0,0,0,2e-3,5e-3;

    Eigen::Vector4d obs_delta = Hxi*delta_pose;


    twc += delta_pose.tail(3); 

    Eigen::Vector3d lPc_delta = Rcw*pw - Rcw*twc;
    Eigen::Vector3d rPc_delta =  filter_ptr->R_rcam_lcam*lPc_delta + filter_ptr->t_rcam_lcam;

    Eigen::Vector4d obs_delta_real;
    obs_delta_real(0) = lPc_delta(0)/lPc_delta(2) - obs.un_u0;
    obs_delta_real(1) = lPc_delta(1)/lPc_delta(2) - obs.un_v0;
    obs_delta_real(2) = rPc_delta(0)/rPc_delta(2) - obs.un_u1;
    obs_delta_real(3) = rPc_delta(1)/rPc_delta(2) - obs.un_v1;

    LOG(INFO) << "obs delta real is " << obs_delta_real.transpose() << 
                "obs delta calculate by jacobian is " << obs_delta.transpose() << endl;
    
    twc -= delta_pose.tail(3);

    // add disturbance to rotation
    Eigen::Matrix<double,6,1> delta_theta;
    delta_theta << 1e-3,3e-3,4e-3,0,0,0;

    Eigen::AngleAxisd theta(delta_theta.head(3).norm(), -delta_theta.head(3)/delta_theta.head(3).norm());

    Eigen::Matrix3d rotate_delta = theta.toRotationMatrix();

    Rcw = rotate_delta*Rcw;

    Eigen::Vector4d obs_delta_theta = Hxi*delta_theta;

    Eigen::Vector3d lPc_delta_theta = Rcw*pw - Rcw*twc;
    Eigen::Vector3d rPc_delta_theta =  filter_ptr->R_rcam_lcam*lPc_delta_theta + filter_ptr->t_rcam_lcam;

    Eigen::Vector4d obs_delta_theta_real;
    obs_delta_theta_real(0) = lPc_delta_theta(0)/lPc_delta_theta(2) - obs.un_u0;
    obs_delta_theta_real(1) = lPc_delta_theta(1)/lPc_delta_theta(2) - obs.un_v0;
    obs_delta_theta_real(2) = rPc_delta_theta(0)/rPc_delta_theta(2) - obs.un_u1;
    obs_delta_theta_real(3) = rPc_delta_theta(1)/rPc_delta_theta(2) - obs.un_v1;

    LOG(INFO) << "obs delta theta real is " << obs_delta_theta_real.transpose() << 
                "obs delta theta calculate by jacobian is " << obs_delta_theta.transpose() << endl;
    
    // add disturbance to feature position
    Rcw = rotate_delta.inverse()*Rcw;
    Eigen::Vector3d delta_w = Eigen::Vector3d(3e-3,2e-3,3e-3);
    pw += delta_w;

    Eigen::Vector4d obs_delta_pw = Hj*delta_w;

    Eigen::Vector3d lPc_delta_pw = Rcw*pw - Rcw*twc;
    Eigen::Vector3d rPc_delta_pw =  filter_ptr->R_rcam_lcam*lPc_delta_pw + filter_ptr->t_rcam_lcam;

    Eigen::Vector4d obs_delta_pw_real;

    obs_delta_pw_real(0) = lPc_delta_pw(0)/lPc_delta_pw(2) - obs.un_u0;
    obs_delta_pw_real(1) = lPc_delta_pw(1)/lPc_delta_pw(2) - obs.un_v0;
    obs_delta_pw_real(2) = rPc_delta_pw(0)/rPc_delta_pw(2) - obs.un_u1;
    obs_delta_pw_real(3) = rPc_delta_pw(1)/rPc_delta_pw(2) - obs.un_v1;

    LOG(INFO) << "obs delta pw real is " << obs_delta_pw_real.transpose() << 
                "obs delta pw calculate by jacobian is " << obs_delta_pw.transpose() << endl;
}
