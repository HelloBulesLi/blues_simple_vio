#pragma once
#include <memory>
#include <condition_variable>
#include "feature_data.hpp"
#include <vector>
#include "cam_state.hpp"
#include "imu_state.hpp"
#include "imu_measure.hpp"
#include "map_feature.hpp"
#include <map>
#include <vector>
#include <deque>
#include <fstream>


namespace vins_slam{
using namespace std;

struct StateServer {
    CamStateSever camera_states;
    IMU_State imu_state;

    // State covariance matrix
    Eigen::MatrixXd state_cov;
    Eigen::Matrix<double, 12, 12> continuos_noise_cov;
};

typedef std::map<int, MapFeature, std::less<int>,
                Eigen::aligned_allocator<std::pair<int, MapFeature>>> MapServer;

typedef std::pair<FeatureInfo_t, std::vector<std::shared_ptr<IMU_Measure>>> Measure_data_set_t;

class OC_MSCKF_Filter {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OC_MSCKF_Filter();
    ~OC_MSCKF_Filter(){};
    void AddImginfo(FeatureInfo_t &img_feature_info);
    void AddImu(std::shared_ptr<IMU_Measure> imu);

    // use the first two hundured imu measurment to init the gravity
    void InitGravity(std::vector<std::shared_ptr<IMU_Measure>> &batch_imu);
    
    bool getMeasure();

    // get imu and feature measure info, mean while align two sensor data
    bool getMeasure_v1();

    // finish the ekf coverance matrix progate, state augmentation, 
    // remove redunant feature, remove redunant cam state, mean while update filter by vision measure
    void process();

    // use batch imu measurement do progate 
    // note, this function is called when receive a new camera frame,
    // every time process a batch imu, increment imu state id 
    void processIMU(double &end_time);

    // progate one imu measure,
    // note that only imu state uncertain have a transition matrix,
    // the cam states transition matrix is an Identity matrix,
    // so only need to calculate the imu state coverance matrix, 
    // then use imu state coverance matrix to calculate the whole state coverance matrix, 
    // the formulation is:
    // [deltaXimu, deltaXcam]^T(progate state) = 
    // [Trans(IMU), Identity(Cam)]*[deltaXimu, deltaXcam]^T(old state) + 
    // Q(only IMU partial state have noise in current implementation)
    void progate(std::shared_ptr<IMU_Measure> new_imu);

    // predict the imu pose use imu measurement
    void predictNewState(const double& dt, const Eigen::Vector3d& gyro,
    const Eigen::Vector3d& acc);

    // cam state augmentation, the state coverance need to increment 6 dimension
    // for a new cam state, the Augmentation operation can express in following formular:
    // [deltaX(k|k) deltaV(k|k)]^T = [Ix J]^T*deltaX(k|k), note the Ix have same dimension
    // with deltaX, the J express the linear approximate quantitative relation of
    // deltaV respect to deltaX

    // note every time when receive new camera frame, augmente a camera state
    // the newest camera state id is the current imu state id 
    void stateAugmentation();

    // add last frame feature info to mapserver( the mapserver include the 
    // all mappoint(landmark/feature point coordinate) observed by last camera frame),
    // the mappoint also record the feature observation info in other camera frame,
    // if the feature have alreadyed in mapserver, the just update the feature observe info,
    // add the current frame observation info to the mappoint,
    // if the feature have not in mapserver, then create a mappoint, add the current frame 
    // observation info to it
    void addNewObservation();


    // remove lost track featur meanwhile use these feature do filter update,
    // note only the feature observed by three camera frame can use to update 
    // the esitmate sate, if the lost track feature's observation num is little
    // than 3, just remove the feature mappoint from mapserver.
    // the filter update operation main step:
    // 1. get the all lost track feature(mappoint) observed by more than three
    // camera frame
    // 2. use above feature(mappoint)'s all observer camera frame and it's pose to get
    // a feature point coordinate estimate in world frame
    // 3. use the lost track feature(mappoint)'s all observes in camera frames
    // to caculate the Jacobain respect to correspondence camera states(Hxi) and world
    // coordinate(Hfj)
    // 4. modify the Hxi and Hfj for consistency of EKF filter
    // 5. use the leftnullspace of Hfj eliminate the measurement update item related to
    // fj(feature coorindate in world frame) , this operation could decrease the 
    // dimension of estimate sate
    // 6. use the whole jacobain of all feature's all observe respect to  update the estimate
    // 7. remove the above features from mapserver
    void removeLostFeature();

    // find the cam state to remove
    void findRedundantCamState(std::vector<unsigned long int> &remove_cam_ids);

    // remove redundant cam state mean while do filter update,
    // when cam states size beyond 30, every time remove two cam states,
    // identify the remove cam states by the parallel distance between two newest
    //  frame and thridth newest frame, if parallel distance  little than setting
    // threlshold, then remove the two newest frame,
    // otherwise remove the oldest cam states. note after filter update, need to 
    // remove the observation of these cam states
    void removeCamState();

    //caculate a feature obs jacobain(multi obs) of 
    // imu state and multi cam_state and feature position,
    // meanwhile caculate the error between real measurement and
    // predict measurement
    void jacobain_calculate(MapFeature &mappoint, std::vector<unsigned long int> &obs_cam_ids,Eigen::MatrixXd &Hxi, 
                            Eigen::MatrixXd &Hfj, Eigen::VectorXd &r);
    
    // calculate the jaobain of a single measure
    void  MeasureJacobian(Cam_State &cam_state, obs_data_t &obs, Eigen::Vector3d &pw, 
                Eigen::Matrix<double,4,6> &Hxi, Eigen::Matrix<double,4,3> &Hj, Eigen::Vector4d &rj);
    
    //do gating test, judge a camera measure is valid
    // reference MINGYang Li paper: https://intra.ece.ucr.edu/~mourikis/papers/Li2013IJRR.pdf
    bool gating_test(Eigen::MatrixXd &Hxi, Eigen::VectorXd &r, uint32_t &dof);
    
    // do filter update operation by the all measurment jacobain 
    // of imu_state and cam state
    void  measurement_update(Eigen::MatrixXd &Hx, Eigen::VectorXd &r);

    // store the traj result
    void publish();

    // for debug
    void publishIMUstate();

    // for debug
    void publishFilterIMUstate();

    // do online reset if the stat_cov is too big
    void OnlineReset();

    // cur state
    StateServer state_server_;

    // extrins between left and right_cam, set when init msckf filter
    Eigen::Matrix3d R_rcam_lcam;
    Eigen::Vector3d t_rcam_lcam;
private:
    std::mutex img_data_mutex_;
    std::condition_variable img_con_;
    std::deque<FeatureInfo_t> img_info_buf_;
    FeatureInfo_t cur_measure_;

    std::mutex data_buf_mutex_;
    std::condition_variable data_buf_con_;
    Measure_data_set_t current_measure_;

    

    double track_rate_;

    // threshold for determine keyframes
    double translation_threshold_;
    double rotation_threshold_;
    double track_rate_threshold_;


    uint32_t max_cam_state_num_;

    // map server
    MapServer map_server_;

    // imu buf
    std::deque<std::shared_ptr<IMU_Measure>> imu_buf_;
    double imu_sample_rate_;
    Eigen::Vector3d gravity_ = Eigen::Vector3d(0,0, -9.81);

    // identify is system initial is finished
    bool is_gravity_set = false;

    Eigen::Matrix3d init_R_rcam_lcam;
    Eigen::Vector3d init_t_rcam_lcam;

    // store traj result
    std::map<unsigned long int, Eigen::Matrix4d, std::less<unsigned long int>,
             Eigen::aligned_allocator<std::pair<unsigned long int, Eigen::Matrix4d>>> traj_result;
    
    std::vector<Eigen::Matrix4d,
            Eigen::aligned_allocator<Eigen::Matrix4d>> imu_traj_result;
    
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> imu_filter_traj_result;

    // observe noise
    double observe_noise;

    // Chi square distribution treshold lookup table for 1-100 dimension,
    // degree of confidence is 0.95
    double chi_sqaure_distribution[100];

    // extrinsic noise
    double extrinsic_rotation_cov;
    double extrinsic_translation_cov;

    // position uncertain threld
    double position_std_threshold;

    bool is_first_img;

    // the count of  no lost track feature
    uint32_t no_lost_track_cnt;

    bool is_select_update_feature = false;

    int online_reset_counter;

    // write the pose after filter update, use evo tool to plot and analyse
    ofstream traj_store;
};

}
