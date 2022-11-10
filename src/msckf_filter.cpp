#include "msckf_filter.hpp"
#include <glog/logging.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseQR>
#include <eigen3/Eigen/SPQRSupport>
#include <boost/math/distributions.hpp>
#include <iostream>


using namespace vins_slam;
using namespace Eigen;
using namespace std;

// set the config param value in constructor function

OC_MSCKF_Filter::OC_MSCKF_Filter()
{
    // set config param
    translation_threshold_ = 0.4;
    rotation_threshold_ = 0.2618;
    track_rate_threshold_ = 0.5;
    max_cam_state_num_ = 20;
    imu_sample_rate_ = 1.0/200;
    gravity_ = {0,0,-9.81};
    is_gravity_set = false;
    position_std_threshold = 8.0;
    online_reset_counter = 0;
    no_lost_track_cnt = 0;
    is_select_update_feature = false;

    // set extrinsic transform form left camera to right camera
    Eigen::Matrix4d left_cam_ext,right_cam_ext;

    left_cam_ext <<  0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0;
    right_cam_ext << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                    0.0, 0.0, 0.0, 1.0;
    Matrix4d T_right_left = right_cam_ext.inverse()*left_cam_ext;
    
    R_rcam_lcam = T_right_left.block<3,3>(0,0);
    t_rcam_lcam = T_right_left.block<3,1>(0,3);

    //set the angle_velocity,bg, velocity, ba noise
    double gyro_noise = 0.005*0.005;
    double acc_noise = 0.05*0.05;
    double gyro_bias_noise = 0.001*0.001;
    double acc_bias_noise = 0.01*0.01;

    state_server_.continuos_noise_cov =
        Matrix<double, 12, 12>::Zero();
    state_server_.continuos_noise_cov.block<3, 3>(0, 0) =
        Matrix3d::Identity()*gyro_noise;
    state_server_.continuos_noise_cov.block<3, 3>(3, 3) =
        Matrix3d::Identity()*gyro_bias_noise;
    state_server_.continuos_noise_cov.block<3, 3>(6, 6) =
        Matrix3d::Identity()*acc_noise;
    state_server_.continuos_noise_cov.block<3, 3>(9, 9) =
        Matrix3d::Identity()*acc_bias_noise;

    observe_noise = 0.035*0.035;



    // set the imu stat_cov initial value
    double gyro_bias_cov = 0.01;
    double acc_bias_cov = 0.01;
    double velocity_cov = 0.25;

    extrinsic_rotation_cov = 3.0462e-4;
    extrinsic_translation_cov = 2.5e-5;

    state_server_.state_cov = MatrixXd::Zero(15,15);

    for (int i = 3; i < 6; i++)
    {
        state_server_.state_cov(i,i) = gyro_bias_cov;
    }

    for (int i = 6; i < 9; i++)
    {
        state_server_.state_cov(i,i) = velocity_cov;
    }

    for (int i = 9; i < 12; i++)
    {
        state_server_.state_cov(i,i) = acc_bias_cov;
    }

    // set imu sate initial param
    state_server_.imu_state.id = 0;
    state_server_.imu_state.timestamp = 0;

    Eigen::Matrix4d Tci = left_cam_ext.inverse();
    state_server_.imu_state.R_cam0_imu = Tci.block<3,3>(0,0);
    state_server_.imu_state.t_imu_cam0 = left_cam_ext.block<3,1>(0,3);

    init_R_rcam_lcam = Tci.block<3,3>(0,0);;
    init_t_rcam_lcam = left_cam_ext.block<3,1>(0,3);

    state_server_.imu_state.rotation_matrix = Eigen::Matrix3d::Identity();

    // initial the chi sqaure lookup table
    for(int i = 1; i < 100; i++)
    {
        boost::math::chi_squared chi_sqaure_dist(i);
        chi_sqaure_distribution[i] = 
                    boost::math::quantile(chi_sqaure_dist, 0.05);
    }

    // clear vector state
    imu_buf_.clear();
    img_info_buf_.clear();

    is_first_img = true;

    // open the file to store imu traj
    traj_store.open("traj_data.tum", ios::out | ios::app);

    if(!traj_store.is_open())
    {
        LOG(ERROR) << "traj_data.tum open failed" << endl;
    }

}

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) {
  Eigen::Matrix3d w_hat;
  w_hat(0, 0) = 0;
  w_hat(0, 1) = -w(2);
  w_hat(0, 2) = w(1);
  w_hat(1, 0) = w(2);
  w_hat(1, 1) = 0;
  w_hat(1, 2) = -w(0);
  w_hat(2, 0) = -w(1);
  w_hat(2, 1) = w(0);
  w_hat(2, 2) = 0;
  return w_hat;
}

inline void quaternionNormalize(Eigen::Vector4d& q) {
  double norm = q.norm();
  q = q / norm;
  return;
}

inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d& q) {
  const Eigen::Vector3d& q_vec = q.block(0, 0, 3, 1);
  const double& q4 = q(3);
  Eigen::Matrix3d R =
    (2*q4*q4-1)*Eigen::Matrix3d::Identity() -
    2*q4*skewSymmetric(q_vec) +
    2*q_vec*q_vec.transpose();
  //TODO: Is it necessary to use the approximation equation
  //    (Equation (87)) when the rotation angle is small?
  return R;
}

inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0;
  Eigen::Vector4d q;
  double dq_square_norm = dq.squaredNorm();

  if (dq_square_norm <= 1) {
    q.head<3>() = dq;
    q(3) = std::sqrt(1-dq_square_norm);
  } else {
    q.head<3>() = dq;
    q(3) = 1;
    q = q / std::sqrt(1+dq_square_norm);
  }

  return q;
}

inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d& R) {
  Eigen::Vector4d score;
  score(0) = R(0, 0);
  score(1) = R(1, 1);
  score(2) = R(2, 2);
  score(3) = R.trace();

  int max_row = 0, max_col = 0;
  score.maxCoeff(&max_row, &max_col);

  Eigen::Vector4d q = Eigen::Vector4d::Zero();
  if (max_row == 0) {
    q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0;
    q(1) = (R(0, 1)+R(1, 0)) / (4*q(0));
    q(2) = (R(0, 2)+R(2, 0)) / (4*q(0));
    q(3) = (R(1, 2)-R(2, 1)) / (4*q(0));
  } else if (max_row == 1) {
    q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0;
    q(0) = (R(0, 1)+R(1, 0)) / (4*q(1));
    q(2) = (R(1, 2)+R(2, 1)) / (4*q(1));
    q(3) = (R(2, 0)-R(0, 2)) / (4*q(1));
  } else if (max_row == 2) {
    q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0;
    q(0) = (R(0, 2)+R(2, 0)) / (4*q(2));
    q(1) = (R(1, 2)+R(2, 1)) / (4*q(2));
    q(3) = (R(0, 1)-R(1, 0)) / (4*q(2));
  } else {
    q(3) = std::sqrt(1+R.trace()) / 2.0;
    q(0) = (R(1, 2)-R(2, 1)) / (4*q(3));
    q(1) = (R(2, 0)-R(0, 2)) / (4*q(3));
    q(2) = (R(0, 1)-R(1, 0)) / (4*q(3));
  }

  if (q(3) < 0) q = -q;
  quaternionNormalize(q);
  return q;
}

inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d& q1,
    const Eigen::Vector4d& q2) {
  Eigen::Matrix4d L;
  L(0, 0) =  q1(3); L(0, 1) =  q1(2); L(0, 2) = -q1(1); L(0, 3) =  q1(0);
  L(1, 0) = -q1(2); L(1, 1) =  q1(3); L(1, 2) =  q1(0); L(1, 3) =  q1(1);
  L(2, 0) =  q1(1); L(2, 1) = -q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
  L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

  Eigen::Vector4d q = L * q2;
  quaternionNormalize(q);
  return q;
}

void OC_MSCKF_Filter::AddImginfo(FeatureInfo_t &img_feature_info)
{
    // if (is_gravity_set)
    // {
    //     img_data_mutex_.lock();
    //     img_info_buf_.push_back(img_feature_info);
    //     img_data_mutex_.unlock();
    //     img_con_.notify_one();
    // }
    if (is_gravity_set)
    {
        // for debug
        LOG(INFO) << "current add  img  timestamp is " << img_feature_info.frame_timestamp << endl;
        data_buf_mutex_.lock();
        img_info_buf_.push_back(img_feature_info);
        data_buf_mutex_.unlock();
        data_buf_con_.notify_one();
    }
    return ;
}

void OC_MSCKF_Filter::AddImu(std::shared_ptr<IMU_Measure> imu)
{
    // imu_buf_.push_back(imu);

    // // LOG(INFO) << "cur imu_buf size is " << imu_buf_.size() << endl;
    // if(!is_gravity_set)
    // {
    //     if(imu_buf_.size() < 200)
    //     {
    //         // LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
    //         return ;
    //     }
    //     LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
    //     std::vector<std::shared_ptr<IMU_Measure>> batch_imu(0);
    //     for(int i = 0; i < 200; i++)
    //     {
    //         // batch_imu.push_back(imu_buf_.front());
    //         batch_imu.push_back(imu_buf_[i]);
    //         // imu_buf_.pop_front();
    //     }
    //     InitGravity(batch_imu);
    //     is_gravity_set = true;
    // }
    data_buf_mutex_.lock();
    imu_buf_.push_back(imu);

    // LOG(INFO) << "cur imu_buf size is " << imu_buf_.size() << endl;
    if(!is_gravity_set)
    {
        if(imu_buf_.size() < 200)
        {
            // LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
            data_buf_mutex_.unlock();
            data_buf_con_.notify_one();
            return ;
        }
        LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
        std::vector<std::shared_ptr<IMU_Measure>> batch_imu(0);
        for(int i = 0; i < 200; i++)
        {
            // batch_imu.push_back(imu_buf_.front());
            batch_imu.push_back(imu_buf_[i]);
            // imu_buf_.pop_front();
        }
        InitGravity(batch_imu);
        imu_buf_.erase(imu_buf_.begin(), imu_buf_.begin()+200);
        is_gravity_set = true;
    }
    data_buf_mutex_.unlock();
    data_buf_con_.notify_one();
}

bool OC_MSCKF_Filter::getMeasure()
{
    if(!img_info_buf_.empty())
    {
        cur_measure_ = img_info_buf_.front();
        img_info_buf_.pop_front();
        LOG(INFO) << "cur img timestamp is " << std::setprecision(16) << 
                    cur_measure_.frame_timestamp << endl;
        return true;
    }
    return false;
}

bool OC_MSCKF_Filter::getMeasure_v1()
{
    if(!img_info_buf_.empty() && !imu_buf_.empty())
    {
        if(imu_buf_.back()->timestamp_ < img_info_buf_.front().frame_timestamp)
        {
            // for debug
            // LOG(INFO) << "wait data for current frame " << endl;
            // LOG(INFO) << "cur imu size is " << imu_buf_.size() << "cur img_buf size is "
            //          << img_info_buf_.size() << std::setprecision(16) << " start imu timestamp is " 
            //          << imu_buf_.begin()->get()->timestamp_ << " last imu timestamp is " 
            //          << imu_buf_.back()->timestamp_  << " oldest img timestamp is "
            //           << img_info_buf_.front().frame_timestamp
            //           << "last img timestamp is " <<
            //            img_info_buf_.back().frame_timestamp << endl;
            return false;
        }

        if(img_info_buf_.back().frame_timestamp < imu_buf_.front()->timestamp_)
        {
            LOG(INFO) << "img data whole delayed for oldest imu data" << endl;
            while(!img_info_buf_.empty())
            {
                img_info_buf_.pop_front();
            }
            return false;
        }

        std::vector<std::shared_ptr<IMU_Measure>> batch_imu;
        // while(imu_buf_.front()->timestamp_ <= img_info_buf_.front().frame_timestamp)
        while(!imu_buf_.empty() && 
              imu_buf_.front()->timestamp_ < img_info_buf_.front().frame_timestamp)
        {
            batch_imu.push_back(imu_buf_.front());
            imu_buf_.pop_front();
        }
        current_measure_ = std::make_pair(img_info_buf_.front(), batch_imu);
        img_info_buf_.pop_front();
        if(!batch_imu.empty())
        {
            LOG(INFO) << "current imu batch start time is "
                      << std::setprecision(16) << batch_imu.front()->timestamp_
                      << " end time is "  << batch_imu.back()->timestamp_ 
                      << " cur img timestamp is " << current_measure_.first.frame_timestamp  << endl;
        }

        return true;
    }
    else
    {
        // for debug
        // LOG(INFO) << "imu_buf_  or img_buf is empty, imu_buf_ size is"
        //         << imu_buf_.size() << " img_buf size is " << img_info_buf_.size() << endl;
    }
    return false;
}

// TODO: reference the result of stereo msckf
void OC_MSCKF_Filter::InitGravity(std::vector<std::shared_ptr<IMU_Measure>> &batch_imu)
{
    Eigen::Vector3d sum_acc_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_angle_vel = Eigen::Vector3d::Zero();
    for(auto imu: batch_imu)
    {
        sum_acc_vel += imu->accelarator;
        sum_angle_vel += imu->angular_vel;
    }

    Eigen::Vector3d mean_acc_vel = sum_acc_vel/batch_imu.size();

    double norm = mean_acc_vel.norm();

    Eigen::Vector3d gravity = {0,0,-norm};

    IMU_State &imu_state = state_server_.imu_state;

    // calculate the gyro bias
    imu_state.bg = sum_angle_vel/batch_imu.size();
    
    Eigen::Quaterniond q_w_i = Quaterniond::FromTwoVectors(mean_acc_vel, -gravity);
    imu_state.rotation_vec = rotationToQuaternion(q_w_i.toRotationMatrix().transpose());
    imu_state.rotation_matrix = quaternionToRotation(imu_state.rotation_vec);

    LOG(INFO) << "the initial rotation q is " << 
                    imu_state.rotation_vec.transpose() << endl <<
                  "the initial rotation matrix is " << imu_state.rotation_matrix << endl;
    
    // imu_state.timestamp = batch_imu.back()->timestamp_;
    
    gravity_ = gravity;

    LOG(INFO) << "the initial gravity is " << std::setprecision(15) << gravity_.transpose() << endl;
}

// call process in a thread, each call do a extend kalman filter operation(include progate and update)
void OC_MSCKF_Filter::process()
{
    if(!is_gravity_set)
    {
        // if(imu_buf_.size() < 200)
        // {
        //     // LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
        //     return ;
        // }
        // LOG(INFO) << "imu_buf size is " << imu_buf_.size() << endl;
        // std::vector<std::shared_ptr<IMU_Measure>> batch_imu(0);
        // for(int i = 0; i < 200; i++)
        // {
        //     batch_imu.push_back(imu_buf_.front());
        //     imu_buf_.pop_front();
        // }
        // InitGravity(batch_imu);
        // is_gravity_set = true;
        return;
    }
    else
    {
        // std::unique_lock<std::mutex> lk(img_data_mutex_);
        // img_con_.wait(lk, [&]{
        //     return getMeasure();
        // });
        // lk.unlock();
        // std::unique_lock<std::mutex> lk(data_buf_mutex_);

        // if(imu_buf_.empty() || img_info_buf_.empty())
        // {
        //     // data_buf_con_.wait(lk, [&]{
        //     //     return getMeasure_v1();
        //     // });
        //     data_buf_con_.wait_until(lk,  std::chrono::steady_clock::now() + chrono::milliseconds(10),
        //                     [&]{ return getMeasure_v1(); });
        //     if(current_measure_.first.frame_feature.empty())
        //     {
        //         if(!getMeasure_v1())
        //         {
        //             lk.unlock();
        //             return ;
        //         }
        //     }
        // }
        // else
        // {
        //     if(!getMeasure_v1())
        //     {
        //         lk.unlock();
        //         return ;
        //     }
        // }

        // lk.unlock();

        // LOG(INFO) << "current msckf measure acquire " << endl;
        std::unique_lock<std::mutex> lk(data_buf_mutex_);
        // data_buf_con_.wait(lk, [&]{
        //     return getMeasure_v1();
        // });
        data_buf_con_.wait_until(lk, std::chrono::system_clock::now() + 10ms);

        if(!getMeasure_v1())
        {
            lk.unlock();
            return ;
        }

        lk.unlock();

        // LOG(INFO) << "current msckf process start " << endl;

        if (is_first_img)
        {
            is_first_img = false;
            state_server_.imu_state.timestamp = current_measure_.first.frame_timestamp;
        }

        // use cur_measure do filter operation

        // progate by batch imu measurement
        processIMU(current_measure_.first.frame_timestamp);

        // for debug, store the imu batch progate traj
        publishIMUstate();
        
        // state augmentation
        stateAugmentation();

        // add new feature obs to mapserver
        addNewObservation();

        // remove lost track feature, mean while do a filter update
        removeLostFeature();

        // for debug, store the imu filter traj
        publishFilterIMUstate();


        // remove redundant cam state and it's obs for keep Calculation complexity, 
        // do a filter update befor remove a cam state 
        removeCamState();

        // publish the pose and feature point cloud for a display,
        // do it in a single thread, and store the result of odometry
        publish();

        // store the imu pose
        IMU_State &imu_state = state_server_.imu_state;
        
        Eigen::Vector4d q_w_i = rotationToQuaternion(imu_state.rotation_matrix.transpose());

        Eigen::Quaterniond q_real(imu_state.rotation_matrix.transpose());

        traj_store << std::setprecision(16) << current_measure_.first.frame_timestamp
                     << ' ' << imu_state.position(0) << ' ' << imu_state.position(1) << ' ' 
                     << imu_state.position(2) << ' ' << q_w_i(0) << ' ' << q_w_i(1) << ' '
                     << q_w_i(2) << ' ' << q_w_i(3) << endl;
        
        LOG(INFO) << "current imu  state timestamp is " << current_measure_.first.frame_timestamp <<
                    "eigen quat is " << q_real.x() << ' ' << q_real.y() << ' ' << q_real.z() << ' ' << q_real.w() << endl;

        // do online reset when estimate coverance matrix beyond the treld
        OnlineReset();
    }    

}

void OC_MSCKF_Filter::processIMU(double &end_time)
{
    //
    int used_imu_cnt = 0;

    // for(auto iter = imu_buf_.begin(); iter != imu_buf_.end(); iter++)
    // {
    //     if(iter->get()->timestamp_ < state_server_.imu_state.timestamp)
    //     {
    //         used_imu_cnt++;
    //         continue;
    //     }

    //     if(iter->get()->timestamp_ >= end_time)
    //     {
    //         break;
    //     }

    //     progate(*iter);
    //     used_imu_cnt++;
    // }

    for(auto imu:current_measure_.second)
    {
        if(imu->timestamp_ < state_server_.imu_state.timestamp)
        {
            continue;
        }

        progate(imu);
        used_imu_cnt++;
    }

    if(used_imu_cnt != 0)
    {
        state_server_.imu_state.id++;
        // set the imu state timestamp for next progate
        // state_server_.imu_state.timestamp = current_measure_.second.back()->timestamp_ + 0.005;

        LOG(INFO) << "cur imu batch start time is " << std::setprecision(16) << 
                   current_measure_.second.begin()->get()->timestamp_ <<
                    "end time is " << state_server_.imu_state.timestamp << endl;
        LOG(INFO) << "real used imu measure is " << used_imu_cnt << endl;
    }

    LOG(INFO) << "state coverance after progate is " << endl << state_server_.state_cov << endl;
}

void OC_MSCKF_Filter::progate(std::shared_ptr<IMU_Measure> new_imu)
{
    IMU_State &imu_state = state_server_.imu_state;
    Eigen::Matrix<double,15,15> F;
    Eigen::Vector3d w = new_imu->angular_vel - imu_state.bg;
    Eigen::Vector3d a = new_imu->accelarator - imu_state.ba;
    double dtime = new_imu->timestamp_ - imu_state.timestamp;
    F.setZero();
    F.block<3,3>(0,0) = -skewSymmetric(w);
    F.block<3,3>(0,3) = -Eigen::Matrix<double,3,3>::Identity();
    F.block<3,3>(6,0) = -imu_state.rotation_matrix.transpose() * skewSymmetric(a);
    F.block<3,3>(6,9) = -imu_state.rotation_matrix.transpose();
    F.block<3,3>(12,6) = Eigen::Matrix3d::Identity();

    // the G matrix used to calculate the progation covarance matrix
    Eigen::Matrix<double,15,12> G;
    
    G.setZero();
    G.block<3,3>(0,0) = -Eigen::Matrix<double,3,3>::Identity();
    G.block<3,3>(3,3) = Eigen::Matrix<double,3,3>::Identity();
    G.block<3,3>(6,6) = -imu_state.rotation_matrix.transpose();
    G.block<3,3>(9,9) = Eigen::Matrix<double,3,3>::Identity();

    Eigen::Matrix<double,15,15> Trans;

    Eigen::Matrix<double,15,15> Fdt = F*dtime;
    Eigen::Matrix<double,15,15> Fdt_square = Fdt*Fdt;
    Eigen::Matrix<double,15,15> Fdt_cube = Fdt_square*Fdt;
    Trans = Eigen::Matrix<double,15,15>::Identity() + Fdt + 0.5*Fdt_square + 1/6*Fdt_cube;

    // for debug
    // LOG(INFO) << "w is " << w.transpose() << "a is " << a.transpose()
    //           << "dt is " << imu_sample_rate_ << endl;

    // predict the imu pose use four order Runge-Kutta 

    // predictNewState(imu_sample_rate_, w, a);
    predictNewState(dtime, w, a);

    // modify the transimation matrix to keep the consistency
    // modify Trans element use position_null, rotation_null, velocity_null

    Eigen::Matrix3d rotation_null_matrix = quaternionToRotation(imu_state.rotation_null);

    Trans.block<3,3>(0,0) = imu_state.rotation_matrix * rotation_null_matrix.transpose();

    Eigen::Matrix3d A = Trans.block<3,3>(6,0);
    
    Eigen::Vector3d u = rotation_null_matrix*gravity_;

    Eigen::Vector3d constrian_w = skewSymmetric(imu_state.velocity_null - 
                                  imu_state.velocity)*gravity_;
    
    Eigen::RowVector3d s = (u.transpose()*u).inverse()*u.transpose();

    Trans.block<3,3>(6,0) = A - (A*u - constrian_w)*s;

    A = Trans.block<3,3>(12,0);

    constrian_w = skewSymmetric(dtime*imu_state.velocity_null + 
                    imu_state.position_null - imu_state.position)*gravity_;
    
    Trans.block<3,3>(12,0) = A - (A*u - constrian_w)*s;


    Eigen::Matrix<double, 15,15> Q;
    Q = Trans*G*state_server_.continuos_noise_cov*G.transpose()*Trans.transpose()*dtime;

    state_server_.state_cov.block<15,15>(0,0) = Trans*state_server_.state_cov.block<15,15>(0,0)*Trans.transpose() + Q;

    // use trans to update the whole state_cov of state_server if 
    // camera states size not zero
    if(state_server_.camera_states.size() > 0)
    {
        uint32_t old_cols = state_server_.state_cov.cols(); 
        uint32_t old_rows = state_server_.state_cov.rows(); 
        state_server_.state_cov.block(0,15,15,old_cols - 15) = Trans*
                                    state_server_.state_cov.block(0,15,15, old_cols - 15);
        state_server_.state_cov.block(15,0, old_rows-15, 15) = 
                                    state_server_.state_cov.block(15,0, old_rows-15, 15)*Trans.transpose();
    }
    
    // ensure p is a symmetric matrix
    Eigen::MatrixXd state_cov_fixed = (state_server_.state_cov + state_server_.state_cov.transpose())/2.0;

    state_server_.state_cov = state_cov_fixed;

    // update the state corresponds to null space
    imu_state.rotation_null = imu_state.rotation_vec;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    imu_state.timestamp = new_imu->timestamp_;
}

void OC_MSCKF_Filter::predictNewState(const double& dt, const Eigen::Vector3d& gyro,
    const Eigen::Vector3d& acc)
{
      // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server_.imu_state.rotation_vec;
  Vector3d& v = state_server_.imu_state.velocity;
  Vector3d& p = state_server_.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc + gravity_;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    gravity_;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    gravity_;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    gravity_;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  state_server_.imu_state.rotation_matrix = quaternionToRotation(state_server_.imu_state.rotation_vec);

  // for debug
    // LOG(INFO) << "current position is " << state_server_.imu_state.position.transpose();

  return;
}


void OC_MSCKF_Filter::stateAugmentation()
{
    // create a new camera state, set the cam state id to imu_state id
    state_server_.camera_states[state_server_.imu_state.id] = 
                                Cam_State(state_server_.imu_state.id);

    // get current camera predict pose by imu predict pose and extrinsic transform from imu to camera
    IMU_State &imu_state = state_server_.imu_state;
    Eigen::Matrix3d R_c_w;
    R_c_w = imu_state.R_cam0_imu*imu_state.rotation_matrix;
    Eigen::Vector3d t_w_c = imu_state.position + 
            imu_state.rotation_matrix.transpose()*imu_state.t_imu_cam0;
    
    Cam_State &cam_state = state_server_.camera_states[imu_state.id];
    cam_state.id = imu_state.id;
    cam_state.timestamp = current_measure_.first.frame_timestamp;
    cam_state.position = t_w_c;
    // cam_state.rotation_matrix = R_c_w;
    cam_state.rotation_vec = rotationToQuaternion(R_c_w);
    cam_state.rotation_matrix = quaternionToRotation(cam_state.rotation_vec);


    // set camera pose null space
    cam_state.position_null = t_w_c;
    cam_state.rotation_null = cam_state.rotation_vec;
    
    // for debug
    LOG(INFO) << "cur imu state id is " << imu_state.id << endl <<std::setprecision(16) <<" cur imu rotation vec is " 
                << imu_state.rotation_vec.transpose() << endl << "cur imu position is " << imu_state.position.transpose() 
                << endl 
                << "cur imu gyro bias is " << imu_state.bg.transpose() << endl
                << "cur imu acc bias is " << imu_state.ba.transpose() << endl
                << "cur imu velocity is " << imu_state.velocity.transpose() << endl
                << "cur imu rotation respect to camera is " << endl<< imu_state.R_cam0_imu << endl
                << "cur cam postion respect to imu is " << imu_state.t_imu_cam0.transpose() << endl
                << "cur imu timestamp is " << imu_state.timestamp << endl;
    
    LOG(INFO) << std::setprecision(16)
                <<"the imu rotation uncertain is " << endl << state_server_.state_cov.block<3,3>(0,0) << endl
                << "the imu position uncertain is " << endl << state_server_.state_cov.block<3,3>(12,12) << endl
                << "the imu velocity uncertain is " << endl << state_server_.state_cov.block<3,3>(6,6) << endl
                << "the imu gyro bias uncertain is " << endl << state_server_.state_cov.block<3,3>(3,3) << endl
                << "the imu acc bias uncertain is " << endl << state_server_.state_cov.block<3,3>(9,9) << endl;

    // for debug
    LOG(INFO) << "the augment camera id is " << cam_state.id << std::setprecision(16) <<
                 " rotation matrix is " << endl << cam_state.rotation_matrix 
               << endl << "rotation vec is" << cam_state.rotation_vec.transpose()
                << "the augment camera position is "<< cam_state.position.transpose() << endl;
    
    // calculate the jacobain of cam state
    Eigen::Matrix<double, 6, 15> J;
    J.setZero();
    J.block<3,3>(0,0) = imu_state.R_cam0_imu;
    // J.block<3,3>(0,15) = Eigen::Matrix3d::Identity();
    J.block<3,3>(3,0) = skewSymmetric(imu_state.rotation_matrix.transpose()*imu_state.t_imu_cam0);
    J.block<3,3>(3,12) = Eigen::Matrix3d::Identity();
    // J.block<3,3>(3,18) = imu_state.rotation_matrix.transpose();

    // resize stat cov matrix
    uint32_t old_rows = state_server_.state_cov.rows();
    uint32_t old_cols = state_server_.state_cov.cols();
    state_server_.state_cov.conservativeResize(old_rows+6, old_cols+6);

    Eigen::Matrix<double,15,15> P11 = state_server_.state_cov.block<15,15>(0,0);
    const Eigen::MatrixXd &P12 = state_server_.state_cov.block(0,15,15,old_cols-15);

    state_server_.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
    state_server_.state_cov.block(0, old_cols, old_rows, 6) = 
                            state_server_.state_cov.block(old_rows,0,6,old_cols).transpose();
    state_server_.state_cov.block(old_rows,old_cols,6,6) = J*P11*J.transpose();

    // ensure the stat cov is symmetric
    MatrixXd state_cov_fixed = (state_server_.state_cov 
                                + state_server_.state_cov.transpose())/2.0;
    
    state_server_.state_cov = state_cov_fixed;

    // for debug
    // LOG(INFO) << "current state uncertain matrix size is " << state_server_.state_cov.rows() 
    //             << ' ' << state_server_.state_cov.cols() << endl;
    
    // LOG(INFO) << "state coverance after augment is " << endl << state_server_.state_cov << endl;

    return;
}

// TODO: check the mappoint feature
void OC_MSCKF_Filter::addNewObservation()
{
    int track_number = 0;
    int cur_feature_number = map_server_.size();

    // for debug, print all map server id
    // for(auto iter = map_server_.begin(); iter != map_server_.end(); iter++)
    // {
    //     LOG(INFO) << "cur map_server id is " << iter->first 
    //               << "cur obs num is "<< iter->second.obs_map.size() << endl; 
    // }

    IMU_State &imu_state = state_server_.imu_state;
    for (auto feat:current_measure_.first.frame_feature)
    {
        obs_data_t obs;
        obs.u0 = feat.u0;
        obs.v0 = feat.v0;
        obs.u1 = feat.u1;
        obs.v1 = feat.v1;
        obs.un_u0 = feat.un_u0;
        obs.un_v0 = feat.un_v0;
        obs.un_u1 = feat.un_u1;
        obs.un_v1 = feat.un_v1;
        if(map_server_.find(feat.feature_id) ==
            map_server_.end())
        {
            // why map_server element can't direct assignment
            // why must define a operator = ?
            // MapFeature map(feat.feature_id);
            // map_server_[feat.feature_id] =  MapFeature(feat.feature_id);
            map_server_[feat.feature_id].feature_id = feat.feature_id;
            map_server_[feat.feature_id].add_observation(imu_state.id, obs);
        }
        else
        {
            track_number++;
            map_server_[feat.feature_id].add_observation(imu_state.id, obs);
        }
    }

    if(cur_feature_number != 0)
    {
        track_rate_ = 1.0*track_number/cur_feature_number*1.0;

        LOG(INFO) << "cure frame track rate is " <<  track_rate_
                    << "cure feature num is " << cur_feature_number << 
                    "cur track number is " << track_number << endl;
        cout << "cure frame track rate is " <<  track_rate_
            << "cure feature num is " << cur_feature_number << 
            "cur track number is " << track_number << endl
            << "cur imu id is " << imu_state.id << " cur imu state timestamp is "
            << std::setprecision(16) << imu_state.timestamp << endl;
    }
}

void OC_MSCKF_Filter::removeLostFeature()
{
    // for debug, print all map server id
    // for(auto iter = map_server_.begin(); iter != map_server_.end(); iter++)
    // {
    //     LOG(INFO) << "cur map_server id is " << iter->first 
    //               << "cur obs num is "<< iter->second.obs_map.size() << endl; 
    // }
    
    std::vector<int> invalid_feature_ids;
    std::vector<int> process_feature_ids;

    IMU_State &imu_state = state_server_.imu_state;

    int jacobain_size = 0;

    for(auto iter = map_server_.begin(); iter != map_server_.end(); iter++)
    {
        if(iter->second.obs_map.find(imu_state.id) != iter->second.obs_map.end())
        {
            continue;
        }

        LOG(INFO) << "cur lost track feature id is " << iter->first 
                << " obs size is " << iter->second.obs_map.size() << endl;

        // if(iter->second.obs_map.size() < 3)
        // if(iter->second.obs_map.size() < 2)
        if(iter->second.obs_map.size() < 3)
        {
            invalid_feature_ids.push_back(iter->first);
            continue;
        }

        // initial the feature position
        if(!iter->second.is_initial)
        {
            if(!iter->second.check_motion(state_server_.camera_states))
            {
                invalid_feature_ids.push_back(iter->first);
                continue;
            }
            Eigen::Matrix4d T_rcam_lcam;
            T_rcam_lcam.setIdentity();
            T_rcam_lcam.block<3,3>(0,0) = R_rcam_lcam;
            T_rcam_lcam.block<3,1>(0,3) = t_rcam_lcam;
            if(!iter->second.initial_position(state_server_.camera_states, T_rcam_lcam))
            {
                invalid_feature_ids.push_back(iter->first);
                continue;
            }
        }


        process_feature_ids.push_back(iter->first);

        jacobain_size += 4*iter->second.obs_map.size() - 3;
    }
    
    LOG(INFO) << "jacobain size befor calculate jacobian is " << jacobain_size << endl;

    CamStateSever &cam_states = state_server_.camera_states;
    Eigen::MatrixXd H_x(jacobain_size, 15+6*cam_states.size());
    // Eigen::MatrixXd H_j(jacobain_size, 3*process_feature_ids.size());
    Eigen::VectorXd r(jacobain_size);

    // remove the little obs feature
    for(auto feat_id:invalid_feature_ids)
    {
        map_server_.erase(feat_id);
    }

    // if have a long time on lost track feature, just select the latest feature to update fitler
    int feature_cnt = 0;
    int select_jacobain_size = 0;

    // if (state_server_.camera_states.size() < max_cam_state_num_ - 2)
    // {
    if(jacobain_size == 0)
    {
        no_lost_track_cnt++;
    }
    else
    {
        no_lost_track_cnt = 0;
    }

    no_lost_track_cnt = 0;

    if(no_lost_track_cnt == 5)
    {
        std::vector<int> init_failed_feature_ids(0);
        for(auto iter = --map_server_.end(); iter != map_server_.begin(); iter--)
        {
            if(iter->second.obs_map.size() >= 3)
            {
                if(!iter->second.is_initial)
                {
                    if(!iter->second.check_motion(state_server_.camera_states))
                    {
                        init_failed_feature_ids.push_back(iter->first);
                        continue;
                    }
                    Eigen::Matrix4d T_rcam_lcam;
                    T_rcam_lcam.setIdentity();
                    T_rcam_lcam.block<3,3>(0,0) = R_rcam_lcam;
                    T_rcam_lcam.block<3,1>(0,3) = t_rcam_lcam;
                    if(!iter->second.initial_position(state_server_.camera_states, T_rcam_lcam))
                    {
                        init_failed_feature_ids.push_back(iter->first);
                        continue;
                    }
                }
                process_feature_ids.push_back(iter->first);
                feature_cnt++;
                select_jacobain_size += 4*iter->second.obs_map.size() - 3;
            }
            if(feature_cnt >= 10)
            {
                break;
            }
        }
        no_lost_track_cnt = 0;
        H_x.resize(select_jacobain_size,  15+6*cam_states.size());
        r.resize(select_jacobain_size);

        // remove the feature can't init position
        for(auto feat_id:init_failed_feature_ids)
        {
            map_server_.erase(feat_id);
        }
    }
    // }

    LOG(INFO) << "used feature cnt for update filter is " << feature_cnt << endl;
    
    // caculate jacobain of every feature, and store it in correspondece position of H_x
    int real_jacobian_size = 0;

    for(auto feat_id:process_feature_ids)
    {
        // get current feature obs cam id vector
        std::vector<unsigned long int> valid_cam_ids;
        for(auto kp : map_server_.find(feat_id)->second.obs_map)
        {
            // identify the camera states already haved in state_servers_'s cam_states
            if(cam_states.find(kp.first) != cam_states.end())
            {
                valid_cam_ids.push_back(kp.first);
            }
            else
            {
                LOG(INFO) << "cure camerastate is not in windows " << kp.first << endl;
            }
        }

        // judge how many obs can use
        uint32_t valid_obs_size = 4*valid_cam_ids.size();
        Eigen::MatrixXd Hxi(valid_obs_size - 3, 
                          15+6*cam_states.size());
        Eigen::MatrixXd Hfj(valid_obs_size, 3);

        Eigen::VectorXd rj(valid_obs_size - 3);

        // calculate the jacobain, and add constraint to null space for consistency
        jacobain_calculate(map_server_.find(feat_id)->second, valid_cam_ids, Hxi, Hfj, rj);

        // do gating test to judge is use this measurement to update filter
        // Stereo MSCKF have error in dof choose of chi square distribution
        // reference the paper of MingYang Li:
        // https://intra.ece.ucr.edu/~mourikis/papers/Li2013IJRR.pdf

        uint32_t obs_size = valid_obs_size - 3;
    
        if(gating_test(Hxi, rj, obs_size))
        {
            // store the Hxi, rj to correspondence position in H_x, r
            H_x.block(real_jacobian_size, 0, valid_obs_size - 3, 15+6*cam_states.size()) = Hxi;
            r.block(real_jacobian_size, 0, valid_obs_size - 3, 1) = rj;

            real_jacobian_size += valid_obs_size - 3;
        }

        //limit the matrix size to guarantee the caculation time not too large
        if(real_jacobian_size >= 1500)
        {
            break;
        }
    }

    LOG(INFO) << "after outlier obs remove, real jacobian size is " << real_jacobian_size << endl;
    // modify the Hx and r size depend the real used measure number
    H_x.conservativeResize(real_jacobian_size, 15+6*cam_states.size());
    r.conservativeResize(real_jacobian_size);

    // use Hx,r do the update of filter
    measurement_update(H_x,r);

    // remove the lost track feature
    if(feature_cnt != 0)
    {
        // for(auto &feat_id: process_feature_ids)
        // {
        //     map_server_.at(feat_id).is_initial = false;
        // }
    }
    else
    {
        for(auto &feat_id: process_feature_ids)
        {
            map_server_.erase(feat_id);
        } 
    }

}

void OC_MSCKF_Filter::removeCamState()
{
    if(state_server_.camera_states.size() <  max_cam_state_num_)
    {
        return ;
    }

    // for debug
    LOG(INFO) << "start remove prun camera state " << endl;

    // find the remove cam_ids by the motion, note the remove_cam_ids sequence is ascending order
    std::vector<unsigned long int> remove_cam_ids(0);
    findRedundantCamState(remove_cam_ids);

    int jacobain_size = 0;

    // use the the removing cam state do a filter update
    for(auto iter = map_server_.begin(); iter != map_server_.end();
            iter++)
    {
        std::vector<unsigned long int> involved_cam_ids;

        for(auto cam_id:remove_cam_ids)
        {
            if(iter->second.obs_map.find(cam_id) !=
                iter->second.obs_map.end())
            {
                involved_cam_ids.push_back(cam_id);
            }
        }

        if(involved_cam_ids.size() == 0)
        {
            continue;
        }
        if(involved_cam_ids.size() == 1)
        {
            iter->second.remove_observation(involved_cam_ids[0]);
            continue;
        }

        // judege a mappoint is initial
        if(!iter->second.is_initial)
        {
            if(!iter->second.check_motion(state_server_.camera_states))
            {
                for(auto dcam_id:involved_cam_ids)
                {
                    iter->second.remove_observation(dcam_id);
                }
                continue;
            }
            Eigen::Matrix4d T_rcam_lcam;
            T_rcam_lcam.setIdentity();
            T_rcam_lcam.block<3,3>(0,0) = R_rcam_lcam;
            T_rcam_lcam.block<3,1>(0,3) = t_rcam_lcam;
            if(!iter->second.initial_position(state_server_.camera_states, T_rcam_lcam))
            {
                for(auto dcam_id:involved_cam_ids)
                {
                    iter->second.remove_observation(dcam_id);
                }
                continue;
            }
        }

        jacobain_size += 4*involved_cam_ids.size() - 3;
    }

    CamStateSever &cam_states = state_server_.camera_states;

    Eigen::MatrixXd Hx(jacobain_size, 15 + 6*cam_states.size());
    Eigen::VectorXd r(jacobain_size);
    Hx.setZero();
    r.setZero();

    uint32_t real_jacobain_size = 0;

    for(auto iter = map_server_.begin(); iter != map_server_.end(); iter++)
    {
        // get current feature obs cam id vector
        std::vector<unsigned long int> valid_cam_ids;
        for(auto cam_id:remove_cam_ids)
        {
            // identify the camera states already haved in state_servers_'s cam_states
            if(iter->second.obs_map.find(cam_id) 
                != iter->second.obs_map.end())
            {
                if(cam_states.find(cam_id) != cam_states.end())
                {
                    valid_cam_ids.push_back(cam_id);
                }
                else
                {
                    LOG(INFO) << "can't find cam state " << cam_id << " in window" << endl;
                }
            }
        }

        if(valid_cam_ids.size() == 0)
        {
            continue;
        }

        // for debug
        if(valid_cam_ids.size() == 1)
        {
            LOG(INFO) << "sing measurement, may result error " << endl;
        }

        Eigen::MatrixXd Hxi(4*valid_cam_ids.size() - 3, 15+6*cam_states.size());
        Eigen::MatrixXd Hj(4*valid_cam_ids.size(), 3);
        Eigen::VectorXd rj(4*valid_cam_ids.size() - 3);
        Hxi.setZero();
        Hj.setZero();
        rj.setZero();

        jacobain_calculate(iter->second, valid_cam_ids, Hxi, Hj, rj);

        // do gating test
        uint32_t dof = rj.size();
        if(gating_test(Hxi, rj, dof))
        {
            // store the Hxi rj to correspondence position of Hx and r
            Hx.block(real_jacobain_size, 0, 4*valid_cam_ids.size() - 3, 15+6*cam_states.size()) = Hxi;
            r.block(real_jacobain_size, 0, 4*valid_cam_ids.size() - 3, 1) = rj;

            real_jacobain_size += 4*valid_cam_ids.size() - 3;
        }
        else
        {
            LOG(INFO) << "current measurement error is too large" << endl;
        }
        // if(!gating_test(Hxi, rj, dof))
        // {
        //     LOG(INFO) << "current measurement error is too large" << endl;
        //     continue;
        // }

        // Hx.block(real_jacobain_size, 0, 4*valid_cam_ids.size() - 3, 15+6*cam_states.size()) = Hxi;
        // r.block(real_jacobain_size, 0, 4*valid_cam_ids.size() - 3, 1) = rj;

        // real_jacobain_size += 4*valid_cam_ids.size() - 3;

        for(auto dcam_id:valid_cam_ids)
        {
            iter->second.remove_observation(dcam_id);
        }
    }

    Hx.conservativeResize(real_jacobain_size, 15+6*cam_states.size());
    r.conservativeResize(real_jacobain_size);

    measurement_update(Hx, r);

    // remove the cam state and it's observe in mapserver
    // for(auto cam_id:remove_cam_ids)
    // {
    //     cam_states.erase(cam_id);
    // }


    // modify the esimate state coverance matrix
    for(auto cam_id:remove_cam_ids)
    {
        int start_index = 15+6*std::distance(cam_states.begin(), cam_states.find(cam_id));
        int end_index = start_index+6;
        int old_rows = state_server_.state_cov.rows();
        int old_cols = state_server_.state_cov.cols();

        if(end_index < old_rows)
        {
            state_server_.state_cov.block(0, start_index, old_rows, old_cols - start_index - 6) = 
                          state_server_.state_cov.block(0, end_index, old_rows, old_cols-end_index);
            state_server_.state_cov.block(start_index, 0, old_rows - start_index - 6, old_cols) = 
                                      state_server_.state_cov.block(end_index, 0, old_rows-end_index, old_cols);
            
            state_server_.state_cov.conservativeResize(old_rows-6, old_cols-6);
        }
        else
        {
            state_server_.state_cov.conservativeResize(old_rows-6, old_cols-6);
        }

        // remove the cam state and it's observe in mapserver
        cam_states.erase(cam_id);
    }
}

void OC_MSCKF_Filter::jacobain_calculate(MapFeature &mappoint, std::vector<unsigned long int> &obs_cam_ids, Eigen::MatrixXd &Hxi, 
                        Eigen::MatrixXd &Hfj, Eigen::VectorXd &r)
{
    std::vector<unsigned long int> valid_cam_ids;
    // judge is the cam_id is in obs
    for(auto cam_id : obs_cam_ids)
    {
        if (mappoint.obs_map.find(cam_id) != mappoint.obs_map.end())
        {
            valid_cam_ids.push_back(cam_id);
        }
    }

    sort(valid_cam_ids.begin(), valid_cam_ids.end());

    CamStateSever &cam_states = state_server_.camera_states;

    Eigen::MatrixXd Hx(4*valid_cam_ids.size(), 15 + 6*cam_states.size());
    Eigen::MatrixXd Hpj(4*valid_cam_ids.size(), 3);
    Eigen::VectorXd error(4*valid_cam_ids.size());

    Hx.setZero();
    Hpj.setZero();
    error.setZero();

    LOG(INFO) << "current mappoint position is " << mappoint.pw.transpose() << endl;
    int index = 0;
    for(auto cam_id : valid_cam_ids)
    {
        Eigen::Matrix<double,4,6> Hxi;
        Eigen::Matrix<double,4,3> Hj;
        Eigen::Vector4d rj;

        // calculate a jacobain for single measurement
        MeasureJacobian(cam_states[cam_id], mappoint.obs_map.at(cam_id), mappoint.pw, 
                        Hxi, Hj, rj);

        // modify the Hxi and Hj for consistency

        Eigen::MatrixXd A = Hxi;
        Eigen::Matrix<double,6,1> u;
        Eigen::Matrix<double,3,3> cam_rotation_null = 
                        quaternionToRotation(cam_states[cam_id].rotation_null);
        u.block<3,1>(0,0) = cam_rotation_null*gravity_;

        u.block<3,1>(3,0) = skewSymmetric(mappoint.pw - 
                                cam_states[cam_id].position_null)*gravity_;

        Eigen::Matrix<double,1,6> s = (u.transpose()*u).inverse()*u.transpose();
        
        Hxi = A - A*u*s;

        Hj = -Hxi.block<4,3>(0,3);

        // store Hxi and Hj to correspondence position in Hx and Hpj
        int index_Hx = 15+6*std::distance(cam_states.begin(), cam_states.find(cam_id));

        int index_Hj = 3*std::distance(cam_states.begin(), cam_states.find(cam_id));

        Hx.block(index*4, index_Hx, 4,6) = Hxi;
        // Hpj.block(index*4, index_Hj, 4,3) = Hj;
        Hpj.block(index*4, 0, 4, 3) = Hj;
        error.block(index*4,0, 4,1) = rj;

        index++;
    }

    // for debug
    // LOG(INFO) << "current measure jacobain is " << endl << Hx << endl;

    // eliminate Hpj by QR decomposition, this operation will change the Hx size
    Eigen::JacobiSVD<MatrixXd> svd_helper(Hpj, Eigen::ComputeFullU | 
                                            Eigen::ComputeThinV);
    
    // get the left null space matrix of Hpj
    MatrixXd A = svd_helper.matrixU().rightCols(4*valid_cam_ids.size() - 3);

    // get the Hx after eliminate the Hpj use it's left null space matrix
    MatrixXd Hx_thin(Hx.rows() - 3, Hx.cols());
    Hx_thin = A.transpose()*Hx;

    VectorXd error_thin(error.rows()-3);
    error_thin = A.transpose()*error;

    Hxi = Hx_thin;
    Hfj = Hpj;
    r  = error_thin;
}

void  OC_MSCKF_Filter::MeasureJacobian(Cam_State &cam_state, obs_data_t &obs, Eigen::Vector3d &pw, 
            Eigen::Matrix<double,4,6> &Hxi, Eigen::Matrix<double,4,3> &Hj, Eigen::Vector4d &rj)
{
    
    Eigen::Matrix3d R_c_w = cam_state.rotation_matrix;
    Eigen::Vector3d t_w_c = cam_state.position;

    // project the world coordinate to current camera frame
    Eigen::Vector3d Pc = R_c_w*pw - R_c_w*t_w_c;
    // Pc /= Pc(2);
    double Z_2 = Pc(2)*Pc(2);

    Eigen::Vector3d rPc = R_rcam_lcam*Pc + t_rcam_lcam;
    double rZ_2 = rPc(2)*rPc(2);

    Eigen::Matrix<double,2,3> J_l_Pc;
    Eigen::Matrix<double,2,3> J_r_Pc;

    J_l_Pc << 1/Pc(2), 0, -Pc(0)/Z_2, 0, 1/Pc(2), -Pc(1)/Z_2;
    J_r_Pc << 1/rPc(2), 0, -rPc(0)/rZ_2, 0, 1/rPc(2), -rPc(1)/rZ_2;

    Eigen::Matrix<double,3, 6> J_lPc_cam;
    Eigen::Matrix<double,3, 6> J_rPc_cam;

    J_lPc_cam.block<3,3>(0,0) = skewSymmetric(Pc);
    J_lPc_cam.block<3,3>(0,3) = -R_c_w;

    J_rPc_cam.block<3,3>(0,0) = R_rcam_lcam*skewSymmetric(Pc);
    J_rPc_cam.block<3,3>(0,3) = -R_rcam_lcam*R_c_w;

    Eigen::Matrix<double,3,3> J_lPc_Pw;
    Eigen::Matrix<double,3,3> J_rPc_Pw;

    J_lPc_Pw = R_c_w;
    J_rPc_Pw = R_rcam_lcam*R_c_w;

    Hxi.block(0, 0, 2, 6) = J_l_Pc*J_lPc_cam;
    Hxi.block(2, 0, 2, 6) = J_r_Pc*J_rPc_cam;
    Hj.block(0,0,2,3) = J_l_Pc*J_lPc_Pw;
    Hj.block(2,0,2,3) = J_r_Pc*J_rPc_Pw;


    rj(0) = obs.un_u0 - Pc(0)/Pc(2);
    rj(1) = obs.un_v0 - Pc(1)/Pc(2);
    rj(2) = obs.un_u1 - rPc(0)/rPc(2);
    rj(3) = obs.un_v1 - rPc(1)/rPc(2);
}

bool OC_MSCKF_Filter::gating_test(Eigen::MatrixXd &Hxi, Eigen::VectorXd &r, uint32_t &dof)
{
    Eigen::MatrixXd P1(Hxi.rows(), Hxi.rows());
    LOG(INFO) << "P1 size " << P1.rows() << ' ' << P1.cols() <<
                " Hxi size " << Hxi.rows() << ' ' << Hxi.cols() <<
                " state cov size is " << state_server_.state_cov.rows() <<
                ' ' << state_server_.state_cov.cols() << endl;
    P1 =  Hxi*state_server_.state_cov*Hxi.transpose();
    Eigen::MatrixXd I(Hxi.rows(), Hxi.rows());
    I.setIdentity();
    Eigen::MatrixXd P2 = observe_noise*I;

    double gamma = r.transpose()*(P1+P2).ldlt().solve(r);

    // Prob: observe noise is too big, debug print
    // cur obs noise too big , error is 461410cur r is -0.193419 000126.79 0-46.6091 00072.327 0-29.2141
    // 0047.0193 0-17.1406 0034.9785 00-13.654 0024.2401 0-8.53677 0019.7276 0-7.3369
    if(gamma > chi_sqaure_distribution[dof])
    {
        LOG(WARNING) << "cur obs noise too big , error is "<< gamma << "cur threlshold is " << chi_sqaure_distribution[dof] <<
                    "cur r is " << r.transpose() << endl;
        return false;
    }
    else
    {
        return true;
    }

}

void OC_MSCKF_Filter::measurement_update(Eigen::MatrixXd &Hx, Eigen::VectorXd &r)
{
    if (Hx.rows() == 0 || r.rows() == 0)
    {
        return ;
    }

    LOG(INFO) << "current state coverance befor update is " << endl << state_server_.state_cov << endl;

    Eigen::MatrixXd Hx_thin;
    Eigen::VectorXd r_thin;
    // do QR decomposition for Hx, to reduce the Hx dimension
    if( Hx.rows() > Hx.cols())
    {
        // HouseholderQR<Eigen::MatrixXd> qr_helper(Hx);
        // MatrixXd Q = qr_helper.householderQ();
        // MatrixXd Q1 = Q.leftCols(21+state_server_.camera_states.size()*6);

        // Hx_thin = Q1.transpose()*Hx;
        // r_thin = Q1.transpose()*r;

        Eigen::SparseMatrix<double> Hx_sparse = Hx.sparseView();

        // perform QR decomposition on H_sparse
        Eigen::SPQR<SparseMatrix<double>> spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(Hx_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * Hx).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        Hx_thin = H_temp.topRows(15 + state_server_.camera_states.size()*6);
        r_thin = r_temp.topRows(15 + state_server_.camera_states.size()*6);
    }
    else
    {
        Hx_thin = Hx;
        r_thin = r;
    }

    Eigen::MatrixXd I(Hx_thin.rows(), Hx_thin.rows());
    I.setIdentity();

    Eigen::MatrixXd S = Hx_thin*state_server_.state_cov*Hx_thin.transpose() + 
              observe_noise*I;
    
    Eigen::MatrixXd K_transpose;

    K_transpose = S.ldlt().solve(Hx_thin*state_server_.state_cov);

    Eigen::MatrixXd K = K_transpose.transpose();

    Eigen::VectorXd delta_x = K*r_thin;

    Eigen::VectorXd delta_x_imu = delta_x.segment<15>(0);

    if(delta_x_imu.segment<3>(6).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0)
    {
        LOG(INFO) << "current vel change " << delta_x_imu.segment<3>(6).transpose() << endl;
        LOG(INFO) << "current position change " << delta_x_imu.segment<3>(12).transpose() << endl;
        LOG(INFO) << "current update change is too large" << endl;
    }

    IMU_State &imu_state = state_server_.imu_state;

    Eigen::Vector4d dq_imu = smallAngleQuaternion(delta_x_imu.head<3>());


    imu_state.rotation_vec = quaternionMultiplication(dq_imu, imu_state.rotation_vec);
    imu_state.rotation_matrix = quaternionToRotation(imu_state.rotation_vec);


    imu_state.position += delta_x_imu.segment<3>(12);
    imu_state.bg += delta_x_imu.segment<3>(3);
    imu_state.ba += delta_x_imu.segment<3>(9);
    // if(delta_x_imu.segment<3>(3).norm() < 1e-3)
    // {
    //     imu_state.bg += delta_x_imu.segment<3>(3);
    // }
    // else
    // {
    //     imu_state.bg += delta_x_imu.segment<3>(3)/10.0;
    //     LOG(INFO) << "curren update gyro bias is too large " << endl;
    // }

    // if(delta_x_imu.segment<3>(9).norm() < 1e-3)
    // {
    //     imu_state.ba += delta_x_imu.segment<3>(9);
    // }
    // else
    // {
    //     imu_state.ba += delta_x_imu.segment<3>(9)/10.0;
    //     LOG(INFO) << "current update acc bias is too large " << endl;
    // }
    imu_state.velocity += delta_x_imu.segment<3>(6);


    LOG(INFO) << "cur gyro bias update is " <<  delta_x_imu.segment<3>(3).norm() << 
                "cur acc update is " << delta_x_imu.segment<3>(9).norm() << endl;

    // update the camera state

    CamStateSever &cam_states = state_server_.camera_states;
    int index = 0;

    for(auto iter = cam_states.begin(); iter != cam_states.end(); iter++)
    { 
        Eigen::Vector4d dq_cam = smallAngleQuaternion(delta_x.segment<3>(15+ 6*index));
        iter->second.rotation_vec = quaternionMultiplication(dq_cam, iter->second.rotation_vec);
        iter->second.rotation_matrix = quaternionToRotation(iter->second.rotation_vec);
        iter->second.position += delta_x.segment<3>(15+6*index+3);
        index++;
    }

    // update the state coverance matrix
    Eigen::MatrixXd IKH(K.rows(), Hx_thin.cols());
    IKH.setIdentity();

    state_server_.state_cov = (IKH - K*Hx_thin)*state_server_.state_cov;

    // ensure the stat cov is symmetric
    MatrixXd state_cov_fixed = (state_server_.state_cov + 
                                state_server_.state_cov.transpose())/2.0;
    state_server_.state_cov = state_cov_fixed;

    // if(angle_update.angle() > 1e-4)
    // {
    //     state_server_.state_cov.block(0,15, 
    //                                      state_server_.state_cov.rows(),6)
    //                         = MatrixXd::Zero(state_server_.state_cov.rows(),6);
    //     state_server_.state_cov.block(15,0, 
    //                                     6,state_server_.state_cov.cols())
    //                         = MatrixXd::Zero(6, state_server_.state_cov.cols());

    //     for (int i = 15; i < 18; i++)
    //     {
    //         state_server_.state_cov(i,i) = extrinsic_rotation_cov;
    //     }

    //     for (int i = 18; i < 21; i++)
    //     {
    //         state_server_.state_cov(i,i) = extrinsic_translation_cov;
    //     }
    // }

    LOG(INFO) << "current state coverance after update is " << endl << state_server_.state_cov << endl;
}


void OC_MSCKF_Filter::publish()
{
    for(auto cam_state:state_server_.camera_states)
    {
        Eigen::Matrix4d Twc;
        Twc.setIdentity();
        Twc.block<3,3>(0,0) = cam_state.second.rotation_matrix.transpose();
        Twc.block<3,1>(0,3) = cam_state.second.position;

        if(traj_result.find(cam_state.second.id) 
            != traj_result.end())
        {
            traj_result[cam_state.second.id] = Twc;
            continue;
        }
        traj_result.insert(std::make_pair(cam_state.second.id, Twc));
    }
}

void OC_MSCKF_Filter::publishIMUstate()
{
    Eigen::Matrix4d Twi;
    Twi.setIdentity();
    Twi.block<3,3>(0,0) = state_server_.imu_state.rotation_matrix.transpose();
    Twi.block<3,1>(0,3) = state_server_.imu_state.position;
    imu_traj_result.push_back(Twi);
}

void OC_MSCKF_Filter::publishFilterIMUstate()
{
    Eigen::Matrix4d Twi;
    Twi.setIdentity();
    Twi.block<3,3>(0,0) = state_server_.imu_state.rotation_matrix.transpose();
    Twi.block<3,1>(0,3) = state_server_.imu_state.position;
    imu_filter_traj_result.push_back(Twi);
}

void OC_MSCKF_Filter::OnlineReset()
{
    if(position_std_threshold <= 0) return;

    double position_x_std = state_server_.state_cov(12,12);
    double position_y_std = state_server_.state_cov(13,13);
    double position_z_std = state_server_.state_cov(14,14);

    if(position_x_std < position_std_threshold &&
        position_y_std < position_std_threshold &&
        position_z_std < position_std_threshold)
    {
        return;
    }

    LOG(WARNING) << "cur position uncertain is " << position_x_std << ' '
                  << position_y_std << ' ' << position_z_std << endl;

    online_reset_counter++;

    // set the imu stat_cov initial value
    double gyro_bias_cov = 0.01;
    double acc_bias_cov = 0.01;
    double velocity_cov = 0.25;

    extrinsic_rotation_cov = 3.0462e-4;
    extrinsic_translation_cov = 2.5e-5;

    state_server_.state_cov = MatrixXd::Zero(15,15);

    for (int i = 3; i < 6; i++)
    {
        state_server_.state_cov(i,i) = gyro_bias_cov;
    }

    for (int i = 6; i < 9; i++)
    {
        state_server_.state_cov(i,i) = velocity_cov;
    }

    for (int i = 9; i < 12; i++)
    {
        state_server_.state_cov(i,i) = acc_bias_cov;
    }

    // clear the mapserver and cam_states;
    map_server_.clear();
    state_server_.camera_states.clear();

    LOG(WARNING) << "online reset complete" << endl;

}

void OC_MSCKF_Filter::findRedundantCamState(std::vector<unsigned long int> &remove_cam_ids)
{
    CamStateSever &cam_states = state_server_.camera_states;
    
    auto reference_iter = cam_states.end();
    for(int i = 0; i < 4; i++)
    {
        reference_iter--;
    }

    auto cam_state_iter = reference_iter;
    cam_state_iter++;
    // auto cam_state_iter = reference_iter++;
    // cam_state_iter++;

    auto first_state_iter = cam_states.begin();

    for(int i = 0; i < 2; i++)
    {
        Eigen::Matrix3d key_rotation = reference_iter->second.rotation_matrix;
    
        double relative_theta_last = 
                Eigen::AngleAxisd(cam_state_iter->second.rotation_matrix*key_rotation.transpose()).angle();
        double relative_trans_last = (reference_iter->second.position - cam_state_iter->second.position).norm();

        if( relative_theta_last < rotation_threshold_ &&
            relative_trans_last < translation_threshold_ &&
            track_rate_ > track_rate_threshold_)
        {
            remove_cam_ids.push_back(cam_state_iter->first);
            cam_state_iter++;
        }
        else
        {
            remove_cam_ids.push_back(first_state_iter->first);
            first_state_iter++;
        }
    }

    sort(remove_cam_ids.begin(), remove_cam_ids.end()); // default sort in ascending order of key

}
