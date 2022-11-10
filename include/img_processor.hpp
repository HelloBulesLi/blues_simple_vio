#pragma once
#include <memory>
#include <iostream>
#include "frame.hpp"
#include <mutex>
#include <condition_variable>
#include "feature_data.hpp"
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <deque>
#include "msckf_filter.hpp"
#include <eigen3/Eigen/Core>
#include <map>

namespace vins_slam {

typedef std::pair<std::shared_ptr<Frame>, std::vector<std::shared_ptr<IMU_Measure>>> Measure_Data_t;

class ImgProcessor{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImgProcessor();
    ~ImgProcessor() {};
    void addframe(std::shared_ptr<Frame> frame);

    // create three level imag pyramids
    void createImagePyramids();
    
    // detect feature on the first frame
    void initframe();
    
    // do stereo match, return the disorted img plane coordinate
    // and  normalized image plane coordinate, and use the extrinsic between
    // the left camera and right camera remove outlier
    void StereoMatch(std::vector<cv::Point2f> &left_kps, std::vector<cv::Point2f> &right_kps, 
                    std::vector<cv::Point2f> &left_un_kps, std::vector<cv::Point2f> &right_un_kps,std::vector<uchar> &inlier_set);
    
    
    // this function used to remove outlier in match feature set, main step is :
    // 1. use two point ransac select in correspondence feature set between two sequential camera frame
    // to get a translate esitmate
    // 2.then use the translate(diff a Scale factor relative a 
    // real translation) to judge the inlier in feature set by epipolar geometry constriant 
    // 3. use the inlier set estimate a more accurate translation, and calculate the 
    // error by epipolar geometry constriant
    // 4. choose the best result in ransac set by inliner number
    void twoPointRansac(std::vector<cv::Point2f> &pre_un_kps, std::vector<cv::Point2f> &cur_un_kps, 
                        Eigen::Matrix3d R_c2_c1, std::vector<uchar> &inliner_set, 
                        double probility, double ransac_threld);
    
    // track the feature in previous frame
    int  trackfeature();

    // add new feature to current frame,
    // if have tracked last feature, need to set mask 
    // befor detect the new feature
    void addNewfeature();

    // remove the redundant feature in every grid by grid max feature number trelshold,
    // and use the lifetime(display the feature have observed by how many frames) to
    // sort the feature in every grid
    void Prunfeature();

    // compare the feature by resonse
    static bool CompareFeatureResponse(FeatureMeta &f1, FeatureMeta &f2)
    {
        return f1.response > f2.response;
    }
    // compare the feature by lifetime
    static bool CompareFeatureLifetime(FeatureMeta &f1, FeatureMeta &f2)
    {
        return f1.life_time > f2.life_time;
    }

    // predict the relative pose between previous and current frame use mean
    // imu angle velocity, note that this function only estimate the rotation
    void predictPose(Eigen::Matrix3d &R_l_c2_c1, Eigen::Matrix3d &R_r_c2_c1);
    
    // use predict relative pose between previous and current frame(only estimate rotation)
    // to predict the feature pos in current frame(disorted img plane coordinate)
    void predictFeaturePose(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_un_kps, 
                            std::vector<cv::Point2f> &cur_kps);
    
    void predictFeaturePose_v1(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_un_kps, 
                        std::vector<cv::Point2f> &cur_kps);
    
    void predictFeaturePose_v2(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_un_kps, 
                        std::vector<cv::Point2f> &cur_kps);
                        
    // for imu measurements, no mutex
    void addimu(std::shared_ptr<IMU_Measure> imu);
    
    // process a new frame,
    // if receive the first frame, call initframe,
    // otherwise track the last frame, then detect new feature,
    // final publish the feature info of current frame to msckf filter thread
    void process();
    
    // get measurment from img_buf
    std::shared_ptr<Frame> get_measurement();

    // get measurement from img_buf and imu buf at same time , make sure align the 
    // img and imu data
    bool get_measurement_v1();
    
    // publish the feature info current frame to msckf filter thread
    void Pubfeature();

    // remove outlier by inlier set
    template <typename T>
    void RemoveOutlier(std::vector<T> &input_v,
                 std::vector<uchar> &inliner, std::vector<T> &output_v);

    // enlarge the value of normalized image plane coordinate, ensure the mean feature norm 
    // in featur match set keep one unit, to avoid the numeric problem in matrix calculate

    void rescalePoints(std::vector<cv::Point2f> &kps1, std::vector<cv::Point2f> &kps2,
                        float &scaling_factor);

    void setMsckf_Filter(std::shared_ptr<OC_MSCKF_Filter> ptr);

    // for debug, display the feature and img
    void display(cv::Mat &img, std::vector<cv::Point2f> &kps);

    // for debug, display the stereo result
    void display_stereo(cv::Mat &left_img, cv::Mat &right_img, 
                        std::vector<cv::Point2f> &left_kps, std::vector<cv::Point2f> &right_kps);
private:
    std::deque<std::shared_ptr<Frame>> img_buf_;
    std::mutex img_data_mutex_;
    std::mutex data_buf_mutex_;
    std::condition_variable img_con_;
    std::condition_variable data_buf_con_;
    std::shared_ptr<GridFeature> pre_grid_feature_;
    std::shared_ptr<GridFeature> cur_grid_feature_;
    int global_feature_id_ = 0;
    uint32_t grid_cols_;
    uint32_t grid_rows_;
    uint32_t grid_height_;
    uint32_t grid_width_;
    uint32_t grid_min_feat_num_;
    uint32_t grid_max_feat_num_;
    uint32_t stereo_threld_;
    uint32_t ransac_threld_;
    uint32_t patch_size;
    uint32_t pyramid_levels;

    std::vector<cv::Mat> prev_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam1_pyramid_;

    std::shared_ptr<Frame> cur_img_measure_;
    std::shared_ptr<Frame> pre_img_measure_;
    Measure_Data_t current_measure_;
    Measure_Data_t previous_measure_;
    std::shared_ptr<OC_MSCKF_Filter> filter_ptr_;
    bool is_first_img_ = true;
    cv::Ptr<cv::Feature2D> detector_;

    int max_feature_;

    std::deque<std::shared_ptr<IMU_Measure>> imu_buf_;
    double imu_sample_interval_;

    std::shared_ptr<Camera> left_cam_;
    std::shared_ptr<Camera> right_cam_;
};

}
