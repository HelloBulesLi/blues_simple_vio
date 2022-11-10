#include "img_processor.hpp"
#include <glog/logging.h>
#include <eigen3/Eigen/Geometry>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace vins_slam;
using namespace cv;

// TODO:code review
// TODO:init config param in constructor function
// TODO:identify the img processor function, could use a google test

ImgProcessor::ImgProcessor()
{
    // set config param init value
    is_first_img_ = true;
    grid_cols_ = 5;
    grid_rows_ = 4;
    global_feature_id_ = 0;
    grid_max_feat_num_ = 4;
    grid_min_feat_num_ = 3;
    stereo_threld_ = 5;
    imu_sample_interval_ = 1.0/200;
    max_feature_ = 200;
    ransac_threld_ = 3;
    patch_size = 15;
    pyramid_levels = 3;

    // create detector
    // double minDistance = 20;
    // int blocksize = 3;
    // detector_ = GFTTDetector::create(max_feature_, 0.01, 
    //                                     minDistance, blocksize, true);
    int threlshold = 10;
    detector_ = FastFeatureDetector::create(threlshold);
    
    // initial the left and right camera
    
    Eigen::Matrix<double,3,3> left_K_;
    left_K_ << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1;

    Eigen::Matrix<double,3,3> right_K_;
    right_K_ << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0 ,1;

    Eigen::Matrix4d left_cam_ext,right_cam_ext;

    left_cam_ext <<  0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0;
    right_cam_ext << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                    0.0, 0.0, 0.0, 1.0;
    
    LOG(INFO) << "left cam ext inverse is " << endl << std::setprecision(16)
              << left_cam_ext.inverse() << endl;
    LOG(INFO) << "right cam ext inverse is " << endl << std::setprecision(16)<<
                right_cam_ext.inverse() << endl;
    
    Eigen::Matrix4d right_left_cam;
    right_left_cam = right_cam_ext.inverse()*left_cam_ext;

    LOG(INFO) << "extrinsic from left to right cam is " << endl << 
                std::setprecision(16) << right_left_cam << endl;
    
    left_cam_ = std::shared_ptr<Camera>(new Camera(left_K_, left_cam_ext, 
                                -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05));
    right_cam_ = std::shared_ptr<Camera>(new Camera(right_K_, right_cam_ext,
                              -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05));
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

inline void randperm(std::vector<int> &index_sequence, int Num, std::vector<int> &index)
{
    if(Num > index_sequence.size())
    {
        LOG(INFO) << "random number great than sequence length" << endl;
        return ;
    }

    std::vector<int> temp = index_sequence;

    random_shuffle(temp.begin(), temp.end());

    index.clear();
    for(int i = 0; i < Num; i++)
    {
        index.push_back(temp[i]);
    }
}

void ImgProcessor::addframe(std::shared_ptr<Frame> frame)
{
    // img_data_mutex_.lock();
    // img_buf_.push_back(frame);
    // img_data_mutex_.unlock();
    // img_con_.notify_one();
    data_buf_mutex_.lock();
    img_buf_.push_back(frame);
    data_buf_mutex_.unlock();
    data_buf_con_.notify_one();
}

void ImgProcessor::addimu(std::shared_ptr<IMU_Measure> imu)
{
    // imu_buf_.push_back(imu);
    data_buf_mutex_.lock();
    imu_buf_.push_back(imu);
    data_buf_mutex_.unlock();
    data_buf_con_.notify_one();
}

std::shared_ptr<Frame> ImgProcessor::get_measurement()
{
    std::shared_ptr<Frame> img_measure;
    img_measure.reset();
    if(!img_buf_.empty())
    {
        img_measure = img_buf_.front();
        img_buf_.pop_front();
    }
    return img_measure;
}

bool ImgProcessor::get_measurement_v1()
{
    std::shared_ptr<Frame> img_measure;
    img_measure.reset();
    if(!img_buf_.empty() && !imu_buf_.empty())
    {
        img_measure = img_buf_.front();
        if(imu_buf_.back()->timestamp_ < img_measure->timestamp_)
        {
            // LOG(WARNING) << "wait imu data for this frame " << endl;
            return false;
        }

        if(img_buf_.back()->timestamp_ < imu_buf_.front()->timestamp_)
        {
            while(!img_buf_.empty())
            {
                img_buf_.pop_front();
            }
            LOG(WARNING) << "img data delay than imu data, throw img, this may happen at \
                            beginning " << endl;
            return false;
        }

        std::vector<std::shared_ptr<IMU_Measure>> imu_batch;
        while(imu_buf_.front()->timestamp_ < img_measure->timestamp_)
        {
            imu_batch.push_back(imu_buf_.front());
            imu_buf_.pop_front();
        }
        current_measure_ = std::make_pair(img_measure, imu_batch);
        img_buf_.pop_front();

        // for debug
        if(!imu_batch.empty())
        {
            LOG(INFO) << "cur batch imu star timestamp is " << std::setprecision(16)<< imu_batch.begin()->get()->timestamp_ << " end timestamp is "
                  << (--imu_batch.end())->get()->timestamp_ <<  " img timestamp is " << img_measure->timestamp_ << endl;
        }
        return true;
    }
    else
    {
        // for debug
        // LOG(INFO) << "cur img or imu buf is empty, imu buf is " << imu_buf_.size()
        //             << " img buf is" << img_buf_.size() << endl;
    }
    return false;
}


void ImgProcessor::process()
{
    // while(true)
    // {
        // std::unique_lock<std::mutex> lk(img_data_mutex_);
        // img_con_.wait(lk, [&]{
        //     cur_img_measure_ = get_measurement();
        //     return (cur_img_measure_.get() != nullptr);
        // });
        // if( current_measure_.first.get())
        // {
        //     LOG(INFO) << "get one img measurement, timestamp is "
        //           << current_measure_.first->timestamp_;
        // }

        // lk.unlock();

        // std::unique_lock<std::mutex> lk(data_buf_mutex_);

        // if(img_buf_.empty() || imu_buf_.empty())
        // {
        //     data_buf_con_.wait_until(lk,  std::chrono::steady_clock::now() + chrono::milliseconds(10), [&]{
        //         return get_measurement_v1();
        //     });

        //     if(current_measure_.first.get() == nullptr)
        //     {
        //         if(!get_measurement_v1())
        //         {
        //             lk.unlock();
        //             return;
        //         }
        //     }
        // }
        // else
        // {
        //     if(!get_measurement_v1())
        //     {
        //         lk.unlock();
        //         return ;
        //     }
        // }
        // lk.unlock();

        // LOG(INFO) << "current img measure acquire " << endl;

        std::unique_lock<std::mutex> lk(data_buf_mutex_);

        data_buf_con_.wait_until(lk,  std::chrono::system_clock::now() + 10ms);

        if(!get_measurement_v1())
        {
            // LOG(INFO) << "cout is current" << endl;
            lk.unlock();
            return;
        }
        lk.unlock();

        LOG(INFO) << "current img process start " << endl;

        createImagePyramids();
        
        if(is_first_img_)
        {
            initframe();
            is_first_img_ = false;
            LOG(INFO) << "img processor initial finished" << endl;
        }
        else
        {
            trackfeature();

            addNewfeature();

            Prunfeature();

            // reset current img feature 
        }

        // pub the current frame feature

        Pubfeature();

        // pre_img_measure_ = cur_img_measure_;
        previous_measure_ = current_measure_;
        current_measure_ = std::make_pair(nullptr, std::vector<std::shared_ptr<IMU_Measure>>(0));

        pre_grid_feature_ = cur_grid_feature_;
        
        std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

        cur_grid_feature_.reset();
        cur_grid_feature_ = std::shared_ptr<GridFeature>(new GridFeature());
        for(uint32_t i = 0; i < grid_cols_*grid_rows_; i++)
        {
                (*cur_grid_feature_)[i] = vector<FeatureMeta>(0);
        }

    // }
}

void ImgProcessor::createImagePyramids()
{
    buildOpticalFlowPyramid(current_measure_.first->left_img_ , curr_cam0_pyramid_, 
                            Size(patch_size, patch_size), pyramid_levels, true, 
                            BORDER_REFLECT_101, BORDER_CONSTANT, false);
    buildOpticalFlowPyramid(current_measure_.first->right_img_, curr_cam1_pyramid_,
                            Size(patch_size, patch_size), pyramid_levels, true,
                            BORDER_REFLECT_101, BORDER_CONSTANT, false);
}

void ImgProcessor::initframe()
{
    grid_height_ = static_cast<int>(current_measure_.first->left_img_.rows/grid_rows_);
    grid_width_ = static_cast<int>(current_measure_.first->left_img_.cols/grid_cols_);

    // detect feature and do stereo match
    std::vector<KeyPoint> left_features;

    detector_->detect(current_measure_.first->left_img_, left_features);

    std::vector<cv::Point2f> left_kps;

    for(auto kp:left_features)
    {
        left_kps.push_back(kp.pt);
    }

    std::vector<cv::Point2f> right_kps;

    std::vector<uchar> inliner_set;
    std::vector<cv::Point2f> left_un_kps;
    std::vector<cv::Point2f> right_un_kps;

   StereoMatch(left_kps, right_kps, left_un_kps, right_un_kps, inliner_set);

   // push the feature to corresponce grid
   cur_grid_feature_ = std::shared_ptr<GridFeature>(new GridFeature());
   for(uint32_t i = 0; i < grid_cols_*grid_rows_; i++)
   {
        (*cur_grid_feature_)[i] = vector<FeatureMeta>(0);
   }

   for(uint32_t i = 0; i < left_kps.size(); i++)
   {
        if(inliner_set[i])
        {
            int grid_c = left_kps[i].x / grid_width_;
            int grid_r = left_kps[i].y / grid_height_;
            int code = grid_r*grid_cols_ + grid_c;
            FeatureMeta feat;
            feat.feature_id = global_feature_id_++;
            // feature coordinate have corrected in StereoMatch
            feat.u0 = left_kps[i].x;
            feat.v0 = left_kps[i].y;
            feat.un_u0 = left_un_kps[i].x;
            feat.un_v0 = left_un_kps[i].y;
            feat.u1 = right_kps[i].x;
            feat.v1 = right_kps[i].y;
            feat.un_u1 = right_un_kps[i].x;
            feat.un_v1 = right_un_kps[i].y;
            feat.life_time = 1;
            feat.response = left_features[i].response;
            (*cur_grid_feature_)[code].push_back(feat);
        }
   }

   // sort every grid by response, and erase the redundancy feature point
   // note that only use reference in for(auto...) can change the value of element 
   for(auto &item: *cur_grid_feature_)
   {
        sort(item.second.begin(), item.second.end(), &ImgProcessor::CompareFeatureResponse);
        if(item.second.size() > grid_max_feat_num_)
        {
            item.second.erase(item.second.begin()+grid_max_feat_num_, item.second.end());
        }
        // for debug
        // LOG(INFO) << "cur grid feature num is " << item.second.size() << " grid treshold" << grid_max_feat_num_ << endl;
   }

   LOG(INFO) << "grid feature size is " << grid_rows_ << ' ' << grid_cols_ << ' ' << (*cur_grid_feature_).size() << endl;

    // for debug
    // choose the inliner feature
    std::vector<cv::Point2f> left_valid_kps(0),right_valid_kps(0);
    for(auto &item: *cur_grid_feature_)
    {
        for(auto kp:item.second)
        {
            left_valid_kps.push_back(cv::Point2f(kp.u0,kp.v0));
            right_valid_kps.push_back(cv::Point2f(kp.u1, kp.v1));
        }
    }

    LOG(INFO) << "feature number after grid remove is " << left_valid_kps.size();
    // display_stereo(current_measure_.first->left_img_, current_measure_.first->right_img_,
    //                 left_valid_kps, right_valid_kps);
    // display(current_measure_.first->right_img_, right_valid_kps);

}

void ImgProcessor::display(cv::Mat &img, std::vector<cv::Point2f> &kps)
{

}

void ImgProcessor::display_stereo(cv::Mat &left_img, cv::Mat &right_img, 
                    std::vector<cv::Point2f> &left_kps, std::vector<cv::Point2f> &right_kps)
{
    cv::Mat stereo_result(left_img.rows, 2*left_img.cols, CV_8UC1);
    cv::Mat left_region = stereo_result(Rect(0, 0, left_img.cols, left_img.rows));
    cv::Mat right_region = stereo_result(Rect(left_img.cols, 0, right_img.cols, right_img.rows));

    left_img.copyTo(left_region);
    right_img.copyTo(right_region);

    Mat stereo_img_show;
    cv::cvtColor(stereo_result, stereo_img_show, CV_GRAY2RGB);
	
    LOG(INFO) << "cur frame feature num is " << left_kps.size() << endl;
    
    for(uint32_t i = 0; i < left_kps.size(); i++)
    {
        cv::Point2f r_kp(right_kps[i].x + left_img.cols, right_kps[i].y);

        cv::line(stereo_img_show, left_kps[i], r_kp, cv::Scalar(0, 255, 0), 1);
        cv::circle(stereo_img_show, left_kps[i], 2, cv::Scalar(0,255,0), 2);
        cv::circle(stereo_img_show, r_kp, 2, cv::Scalar(0,255,0), 2);
    }

    imshow("stereo result", stereo_img_show);
}

int ImgProcessor::trackfeature()
{
    grid_width_ = current_measure_.first->left_img_.cols/grid_cols_;
    grid_height_ = current_measure_.first->left_img_.rows/grid_rows_;

    Eigen::Matrix3d R_l_c2_c1, R_r_c2_c1;
    
    R_l_c2_c1.setIdentity();
    R_r_c2_c1.setIdentity();
    predictPose(R_l_c2_c1, R_r_c2_c1);

    // get pre feature info
    std::vector<cv::Point2f> pre_left_pts;
    std::vector<cv::Point2f> pre_left_un_kps;
    std::vector<cv::Point2f> pre_right_un_kps;
    std::vector<int> pre_ids;
    std::vector<int> pre_lifetime;

    for(auto grid_kps: *pre_grid_feature_)
    {
        for(auto kp: grid_kps.second)
        {
            pre_ids.push_back(kp.feature_id);
            pre_lifetime.push_back(kp.life_time);
            pre_left_pts.push_back(cv::Point2f(kp.u0, kp.v0));
            pre_left_un_kps.push_back(cv::Point2f(kp.un_u0, kp.un_v0));
            pre_right_un_kps.push_back(cv::Point2f(kp.un_u1, kp.un_v1));
        }   
    }

    // from the debug show in first 2 senconds, the predict result have not big help,
    // because no enough motion?
    // by the erouc dataset, identi predict feature position by predict rotate is not good,
    // even in a little motion scene, the tracked match is not good
    // when test in real scene, increment the motion then check the track result in two frames

    std::vector<cv::Point2f> left_kps;
    predictFeaturePose(R_l_c2_c1, pre_left_un_kps, left_kps);
    // predictFeaturePose_v1(R_l_c2_c1, pre_left_pts, left_kps);
    // predictFeaturePose_v2(R_l_c2_c1, pre_left_pts, left_kps);

    // for debug
    // display_stereo(previous_measure_.first->left_img_, current_measure_.first->left_img_,
    //             pre_left_pts, left_kps);

    vector<uchar> status;
    cv::Mat error;

    // use LK to track the feature, remove the outlier
    cv::calcOpticalFlowPyrLK(prev_cam0_pyramid_, curr_cam0_pyramid_, pre_left_pts, left_kps, status, error, Size(15,15), 3, 
                                TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                cv::OPTFLOW_USE_INITIAL_FLOW);
    
    for(uint32_t i = 0; i < pre_left_pts.size(); i++)
    {
        if(status[i] == 0) continue;
        if (left_kps[i].y < 0 ||
            left_kps[i].y >  current_measure_.first->left_img_.rows-1||
            left_kps[i].x < 0 ||
            left_kps[i].x > current_measure_.first->left_img_.cols-1)
        {
            status[i] = 0;
        }
    }

    std::vector<cv::Point2f> pre_left_pts_track;
    std::vector<cv::Point2f> pre_left_un_kps_track;
    std::vector<cv::Point2f> pre_right_un_kps_track;
    std::vector<cv::Point2f> left_kps_track;
    std::vector<int> pre_ids_track;
    std::vector<int> pre_lifetime_track;

    RemoveOutlier(left_kps, status, left_kps_track);
    RemoveOutlier(pre_left_pts, status, pre_left_pts_track);
    RemoveOutlier(pre_left_un_kps, status, pre_left_un_kps_track);
    RemoveOutlier(pre_right_un_kps, status, pre_right_un_kps_track);
    RemoveOutlier(pre_ids, status, pre_ids_track);
    RemoveOutlier(pre_lifetime, status, pre_lifetime_track);

    // for debug
    // display_stereo(previous_measure_.first->left_img_, current_measure_.first->left_img_,
    //             pre_left_pts_track, left_kps_track);

    // stereo match, and use fundamental matrix to remove outlier
    
    std::vector<cv::Point2f> right_kps_track;
    std::vector<cv::Point2f> left_un_kps,right_un_kps;
    std::vector<cv::Point2f> left_un_kps_match,right_un_kps_match;
    std::vector<uchar> inlier_set;

    StereoMatch(left_kps_track, right_kps_track, left_un_kps, right_un_kps, inlier_set);

    std::vector<cv::Point2f> pre_left_pts_match;
    std::vector<cv::Point2f> pre_left_un_kps_match;
    std::vector<cv::Point2f> pre_right_un_kps_match;
    std::vector<cv::Point2f> left_kps_match;
    std::vector<cv::Point2f> right_kps_match;
    std::vector<int> pre_ids_match;
    std::vector<int> pre_lifetime_match;

    RemoveOutlier(left_kps_track, inlier_set, left_kps_match);
    RemoveOutlier(right_kps_track, inlier_set, right_kps_match);
    RemoveOutlier(left_un_kps, inlier_set, left_un_kps_match);
    RemoveOutlier(right_un_kps, inlier_set, right_un_kps_match);
    RemoveOutlier(pre_left_pts_track, inlier_set, pre_left_pts_match);
    RemoveOutlier(pre_left_un_kps_track, inlier_set, pre_left_un_kps_match);
    RemoveOutlier(pre_right_un_kps_track, inlier_set, pre_right_un_kps_match);
    RemoveOutlier(pre_ids_track, inlier_set, pre_ids_match);
    RemoveOutlier(pre_lifetime_track, inlier_set, pre_lifetime_match);

    // for debug
    // display_stereo(previous_measure_.first->left_img_, current_measure_.first->left_img_,
    //             pre_left_pts_match, left_kps_match);

    std::vector<uchar> left_ransac_inliner,right_ransac_inliner;

    // use two ransac remove outlier
    twoPointRansac(pre_left_un_kps_match, left_un_kps_match, 
                        R_l_c2_c1, left_ransac_inliner, 0.99, ransac_threld_);
    
    twoPointRansac(pre_right_un_kps_match, right_un_kps_match, R_r_c2_c1, right_ransac_inliner, 0.99, ransac_threld_);

    int after_ransac = 0;
    // push the featur to correspond grid
    std::vector<cv::Point2f> left_kps_dis;
    std::vector<cv::Point2f> pre_left_kps_dis;
    for(uint32_t i = 0; i < left_un_kps_match.size(); i++)
    {
        if(!left_ransac_inliner[i] || !right_ransac_inliner[i])
        {
            continue;
        }

        int grid_c = left_kps_match[i].x/grid_width_;
        int grid_r = left_kps_match[i].y/grid_height_;

        int code = grid_r*grid_cols_ + grid_c;


        FeatureMeta feat;
        feat.feature_id = pre_ids_match[i];
        feat.life_time = pre_lifetime_match[i] + 1;
        feat.u0 = left_kps_match[i].x;
        feat.v0 = left_kps_match[i].y;
        feat.un_u0 = left_un_kps_match[i].x;
        feat.un_v0 = left_un_kps_match[i].y;
        feat.u1 = right_kps_match[i].x;
        feat.v1 = right_kps_match[i].y;
        feat.un_u1 = right_un_kps_match[i].x;
        feat.un_v1 = right_un_kps_match[i].y;

        (*cur_grid_feature_)[code].push_back(feat);

        left_kps_dis.push_back(cv::Point2f(feat.u0, feat.v0));
        pre_left_kps_dis.push_back(pre_left_pts_match[i]);
        after_ransac++;
    }

    double track_rate = 1.0*left_kps_dis.size()/pre_left_pts.size();
    // for debug
    display_stereo(previous_measure_.first->left_img_, current_measure_.first->left_img_,
                pre_left_kps_dis, left_kps_dis);

    // waitKey(0);

    LOG(INFO) << "the track number after ransac is " << after_ransac << endl;
    LOG(INFO) << "the track rate after ransac is " << track_rate << endl;
    return after_ransac;
}

void ImgProcessor::addNewfeature()
{
    grid_height_ = static_cast<int>(current_measure_.first->left_img_.rows/grid_rows_);
    grid_width_ = static_cast<int>(current_measure_.first->left_img_.cols/grid_cols_);
    // set mask for tracked feature
    cv::Mat mask(current_measure_.first->left_img_.size(), CV_8UC1, 255);
    int track_feature_num = 0;
    for(auto grid_kps: *cur_grid_feature_)
    {
        for(auto kp : grid_kps.second)
        {
            // anlanise the error of following code,how to result in crash
            // const int y = static_cast<int>(kp.v0);
            // const int x = static_cast<int>(kp.u0);

            // int up_lim = y-2, bottom_lim = y+3,
            //     left_lim = x-2, right_lim = x+3;
            // if (up_lim < 0) 
            // {
            //     up_lim = 0;
            // }
            // if (bottom_lim >= mask.rows)
            // {
            //     bottom_lim = mask.rows - 1;
            // }
            // if (left_lim < 0)
            // {
            //     left_lim = 0;
            // }
            // if (right_lim >= mask.cols)
            // {
            //     right_lim = mask.cols - 1;
            // }

            // Range row_range(up_lim, bottom_lim);
            // Range col_range(left_lim, right_lim);
            // mask(row_range, col_range) = 0;

            // cv::circle(mask, cv::Point2f(kp.u0,kp.v0), 20, 0, -1);
            cv::circle(mask, cv::Point2f(kp.u0, kp.v0), 3, 0, -1);
            track_feature_num++;
        }
    }

    // detect new feature
    std::vector<cv::KeyPoint> new_feature;
    // detector_->setMaxFeatures(max_feature_ - track_feature_num);
    detector_->detect(current_measure_.first->left_img_, new_feature, mask);


    // remove redundant feature in grid
    std::shared_ptr<GridFeature> new_feature_seive(new GridFeature());
    for(uint32_t i = 0; i < new_feature.size(); i++)
    {
        int grid_c = new_feature[i].pt.x/grid_width_;
        int grid_r = new_feature[i].pt.y/grid_height_;

        int code = grid_r*grid_cols_ + grid_c;
        FeatureMeta kp;
        // kp.feature_id = global_feature_id_++;
        kp.life_time = 1;
        kp.response = new_feature[i].response;
        kp.u0 = new_feature[i].pt.x;
        kp.v0 = new_feature[i].pt.y;

        (*new_feature_seive)[code].push_back(kp);
    }

    for(auto &grid_kps:*new_feature_seive)
    {
        sort(grid_kps.second.begin(), grid_kps.second.end(), &ImgProcessor::CompareFeatureResponse);
        if(grid_kps.second.size() > grid_max_feat_num_)
        {
             grid_kps.second.erase(grid_kps.second.begin()+grid_max_feat_num_, 
                                    grid_kps.second.end());
        }
    }

    // do stereo match
    std::vector<cv::Point2f> left_kps,right_kps;
    std::vector<cv::Point2f> left_un_kps,right_un_kps;
    std::vector<uchar> inlier_set;
    std::vector<float> feature_response;

    for(auto grid_kps: *new_feature_seive)
    {
        for(auto kp : grid_kps.second)
        {
            left_kps.push_back(cv::Point2f(kp.u0,kp.v0));
            feature_response.push_back(kp.response);
        }
    }
    
    StereoMatch(left_kps, right_kps, left_un_kps, right_un_kps, inlier_set);

    // for debug
    // choose the inliner feature
    std::vector<cv::Point2f> left_valid_kps(0),right_valid_kps(0);
    for(uint32_t i = 0; i < left_kps.size(); i++)
    {
        if(inlier_set[i])
        {
            left_valid_kps.push_back(left_kps[i]);
            right_valid_kps.push_back(right_kps[i]);
        }
    }


    // display_stereo(current_measure_.first->left_img_, current_measure_.first->right_img_,
    //                 left_valid_kps, right_valid_kps);


    std::shared_ptr<GridFeature> grid_new_feature(new GridFeature());
    for(uint32_t i = 0; i < grid_cols_*grid_rows_; i++)
    {
            (*grid_new_feature)[i] = vector<FeatureMeta>(0);
    }

    for(uint32_t i = 0; i < left_kps.size(); i++)
    {
        if(inlier_set[i])
        {
            int grid_c = left_kps[i].x/grid_width_;
            int grid_r = left_kps[i].y/grid_height_;

            int code = grid_r*grid_cols_ + grid_c;
            FeatureMeta kp;
            kp.response = feature_response[i];
            kp.u0 = left_kps[i].x;
            kp.v0 = left_kps[i].y;
            kp.u1 = right_kps[i].x;
            kp.v1 = right_kps[i].y;
            kp.un_u0 = left_un_kps[i].x;
            kp.un_v0 = left_un_kps[i].y;
            kp.un_u1 = right_un_kps[i].x;
            kp.un_v1 = right_un_kps[i].y;

            (*grid_new_feature)[code].push_back(kp);
        }
    }

    for(auto &grid_kps:*grid_new_feature)
    {
        sort(grid_kps.second.begin(), grid_kps.second.end(), &ImgProcessor::CompareFeatureResponse);
    }


    // add new feature to correspondence grid,
    // mean while judge the feature num is too many
    int new_feature_num = 0;
    for(uint32_t i = 0; i < grid_cols_*grid_rows_; i++)
    {
        // LOG(INFO) << "cur grid feature num is " << (*cur_grid_feature_)[i].size() 
        //     << " grid new feature num is " << (*grid_new_feature)[i].size();
        if((*cur_grid_feature_)[i].size() >= grid_max_feat_num_)
        {
            continue;
        }

        for(uint32_t k = 0; (k < (*grid_new_feature)[i].size()) && 
                    (k < grid_max_feat_num_ - (*cur_grid_feature_)[i].size()); k++)
        {
            (*grid_new_feature)[i][k].feature_id = global_feature_id_++;
            (*grid_new_feature)[i][k].life_time = 1;
            (*cur_grid_feature_)[i].push_back((*grid_new_feature)[i][k]);
            new_feature_num++;
        }
    }

    LOG(INFO) << "new feature num is " << new_feature_num << endl;
}

void ImgProcessor::Prunfeature()
{
    for(auto &grid_kps:*cur_grid_feature_)
    {
        if(grid_kps.second.size() <= grid_max_feat_num_)
        {
            continue;
        }
        sort(grid_kps.second.begin(), grid_kps.second.end(), &ImgProcessor::CompareFeatureLifetime);
        grid_kps.second.erase(grid_kps.second.begin()+grid_max_feat_num_, grid_kps.second.end());
    }
}

void ImgProcessor::Pubfeature()
{
    std::vector<FeatureMeta> cur_features;
    for(auto grid: *cur_grid_feature_)
    {
        for(auto kp:grid.second)
        {
            cur_features.push_back(kp);
        }
    }

    FeatureInfo pub_features;
    for(auto kp : cur_features)
    {
        pub_features.frame_feature.push_back(kp);
        // for debug
        // LOG(INFO) << "cur feature id is " << kp.feature_id
        //           << "cur timestamp is " << std::setprecision(16) <<
        //           current_measure_.first->timestamp_<< 
        //           "cur life time is " << kp.life_time << endl;
    }

    pub_features.frame_timestamp = current_measure_.first->timestamp_;

    filter_ptr_->AddImginfo(pub_features);
}

void ImgProcessor::StereoMatch(std::vector<cv::Point2f> &left_kps, std::vector<cv::Point2f> &right_kps, 
                std::vector<cv::Point2f> &left_un_kps, std::vector<cv::Point2f> &right_un_kps,std::vector<uchar> &inlier_set)
{
    // predict the right_kps pos use extrinsic between left camera and right camera
    Eigen::Matrix4d T_r_l = right_cam_->ext_cam_imu_ * left_cam_->ext_cam_imu_.inverse();

    for(auto kps: left_kps)
    {
        Eigen::Vector3d gray_dir;
        Eigen::Vector2d img_pos(kps.x,kps.y);
        left_cam_->unproject(img_pos, gray_dir);
        cv::Point2f un_img_pos(gray_dir(0), gray_dir(1));
        left_un_kps.push_back(un_img_pos);

        Eigen::Vector3d right_dir;
        right_dir = T_r_l.block<3,3>(0,0)*gray_dir;
        Eigen::Vector2d right_img_pos;
        right_cam_->project(right_dir, right_img_pos);

        right_kps.push_back(cv::Point2f(right_img_pos(0), right_img_pos(1)));
    }

    vector<uchar> status;
    cv::Mat error;

    cv::calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_, left_kps, right_kps, status, error, Size(15,15), 3, 
                                TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                cv::OPTFLOW_USE_INITIAL_FLOW);
    
    for(uint32_t i = 0; i < left_kps.size(); i++)
    {
        if(status[i] == 0) continue;
        if (right_kps[i].y < 0 ||
            right_kps[i].y >  current_measure_.first->right_img_.rows-1||
            right_kps[i].x < 0 ||
            right_kps[i].x > current_measure_.first->right_img_.cols-1)
        {
            status[i] = 0;
        }
    }

    uint32_t stereo_match_num = 0;

    for(uint32_t i = 0; i < right_kps.size(); i++)
    {
        if(status[i])
        {
            Eigen::Vector3d gray_dir;
            Eigen::Vector2d img_pos(right_kps[i].x, right_kps[i].y);
            right_cam_->unproject(img_pos, gray_dir);

            right_un_kps.push_back(cv::Point2f(gray_dir(0), gray_dir(1)));
            inlier_set.push_back(1);
            stereo_match_num++;
        }
        else
        {
            right_un_kps.push_back(cv::Point2f(0,0));
            inlier_set.push_back(0);
        }
    }

    LOG(INFO) << "the inliner number befor stereo match is "<< left_kps.size() << endl;
    LOG(INFO) << "the inliner number after stereo match and befor fundamental check is "<< stereo_match_num << endl;
    // use the foundamental matrix remove outlier

    Eigen::Matrix3d E;
    E = skewSymmetric(T_r_l.block<3,1>(0,3))*T_r_l.block<3,3>(0,0);
    double norm_pixel_unit = 4/(right_cam_->fx_ + right_cam_->fy_ + left_cam_->fx_ + left_cam_->fy_);

    uint32_t fundamental_check_num = 0;

    for(uint32_t i = 0; i < right_kps.size(); i++)
    {
        if(inlier_set[i])
        {
            Eigen::Vector3d ep_line = E*Eigen::Vector3d(left_un_kps[i].x,left_un_kps[i].y,1);
            Eigen::Vector3d match_pos(right_un_kps[i].x,right_un_kps[i].y,1);
            double error = fabs(match_pos.transpose()*ep_line)/
                                    sqrt(ep_line(0)*ep_line(0) + ep_line(1)*ep_line(1));
            if(error > stereo_threld_ * norm_pixel_unit)
            {
                inlier_set[i] = false;
            }
            else
            {
                fundamental_check_num++;
            }
        }
    }

    LOG(INFO) << "the inliner number after fundamental check is "<< fundamental_check_num << endl;
}

void ImgProcessor::predictPose(Eigen::Matrix3d &R_l_c2_c1, Eigen::Matrix3d &R_r_c2_c1)
{
    Eigen::Vector3d sum_angle_vel;
    sum_angle_vel.setZero();
    double dt = 0;
    int interg_cnt = 0;

    for(auto iter: current_measure_.second)
    {
        if(iter->timestamp_ < previous_measure_.first.get()->timestamp_)
        {
            continue;
        }
        sum_angle_vel += iter->angular_vel;
        dt += imu_sample_interval_;
        interg_cnt++;
    }


    Eigen::Vector3d mean_angle_vel = sum_angle_vel/interg_cnt;

    Eigen::Vector3d left_cam_angle_vel = left_cam_->ext_cam_imu_.block<3,3>(0,0)*mean_angle_vel;
    Eigen::Vector3d right_cam_angle_vel = right_cam_->ext_cam_imu_.block<3,3>(0,0)*mean_angle_vel;

    left_cam_angle_vel *= dt;
    right_cam_angle_vel *= dt;

    Eigen::AngleAxisd left_Axis(left_cam_angle_vel.norm(), left_cam_angle_vel/left_cam_angle_vel.norm());
    Eigen::AngleAxisd right_Axis(right_cam_angle_vel.norm(), right_cam_angle_vel/right_cam_angle_vel.norm());

    R_l_c2_c1 = left_Axis.toRotationMatrix().transpose();
    R_r_c2_c1 = right_Axis.toRotationMatrix().transpose();

    LOG(INFO) << "cur rotate angle is " << left_cam_angle_vel.norm();
}

void ImgProcessor::predictFeaturePose(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_un_kps, 
                            std::vector<cv::Point2f> &cur_kps)
{
    std::vector<cv::Point2f> cur_un_kps;
    for(auto kp: pre_un_kps)
    {
        Eigen::Vector3d cur_kp = R_c2_c1*Eigen::Vector3d(kp.x,kp.y,1);
        Eigen::Vector2d cur_img_pos;
        left_cam_->project(cur_kp, cur_img_pos);
        cur_kps.push_back(cv::Point2f(cur_img_pos(0), cur_img_pos(1)));
    }
}

void ImgProcessor::predictFeaturePose_v1(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_kps, 
                            std::vector<cv::Point2f> &cur_kps)
{
    Eigen::Matrix3d K;
    K.setZero();
    K << left_cam_->fx_, 0, left_cam_->cx_, 0, left_cam_->fy_, left_cam_->cy_, 0, 0,1;
    Eigen::Matrix3d H;

    H = K*R_c2_c1*K.inverse();

    std::vector<cv::Point2f> cur_un_kps;
    for(auto kp: pre_kps)
    {
        Eigen::Vector3d cur_kp = H*Eigen::Vector3d(kp.x,kp.y,1);
        cur_kps.push_back(cv::Point2f(cur_kp(0)/cur_kp(2), cur_kp(1)/cur_kp(2)));
    }
}

void ImgProcessor::predictFeaturePose_v2(Eigen::Matrix3d &R_c2_c1, std::vector<cv::Point2f> &pre_kps, 
                            std::vector<cv::Point2f> &cur_kps)
{
    for(auto kp: pre_kps)
    {
        cur_kps.push_back(kp);
    }
}


void ImgProcessor::twoPointRansac(std::vector<cv::Point2f> &pre_un_kps, std::vector<cv::Point2f> &cur_un_kps, 
                    Eigen::Matrix3d R_c2_c1,std::vector<uchar> &inliner_set,
                    double probility, double ransac_threld)
{
    // generate diffrent index in one time:
    // https://www.cnblogs.com/salan668/p/3652532.html
    // x2^T * t_c2_c1^*R_c2_c1* x1 = 0

    inliner_set.clear();
    inliner_set.resize(cur_un_kps.size(), 1);

    int iter_max = ceil(log(1 - probility)/log(1 - 0.7*0.7));
    double normal_unit_pixel = 2.0/(left_cam_->fx_ + left_cam_->fy_);

    // normalized the point coordinate, TODO: not completely understand ?
    // rescale for numerical problem ?
    std::vector<cv::Point2f> kps1 = pre_un_kps;
    std::vector<cv::Point2f> kps2 = cur_un_kps;
    float scale_factor;
    rescalePoints(kps1, kps2, scale_factor);
    normal_unit_pixel *= scale_factor;

    LOG(INFO) << "scale factor is "<< scale_factor << " normal_unit_pixel is " << normal_unit_pixel;
    LOG(INFO) << "kps coordinate after rescale " << kps1[0].x << ' ' << kps1[0].y << ' ' << kps2[kps2.size() - 1].x
              << ' ' << kps2[kps2.size() - 1].y << endl;
    

    // get the diff bwteen the pre frame and cur frame feature set
    Eigen::MatrixXd diff_set(cur_un_kps.size(), 3);

    for(uint32_t i = 0; i < cur_un_kps.size(); i++)
    {
        Eigen::Vector3d p(kps1[i].x, kps1[i].y, 1);
        Eigen::Vector3d p1 = R_c2_c1*p;
        p1 /= p1(2);
        diff_set(i,0) = p1(1) - kps2[i].y;
        diff_set(i,1) = -p1(0) + kps2[i].x;
        diff_set(i,2) = kps2[i].y*p1(0) - kps2[i].x*p1(1);
    }

    // mark the feature with large motion as outlier distance between seq frames
    // geater than 50 pixel
    std::vector<cv::Point2f> kps_diff(cur_un_kps.size());
    for(uint32_t i = 0; i < kps_diff.size(); i++)
    {
        kps_diff[i] = kps1[i] - kps2[i];
    }

    double mean_pt_dis = 0;
    int raw_inlier_cnt = 0;
    for (uint32_t i = 0; i < kps_diff.size(); i++)
    {
        double dis = sqrt(kps_diff[i].dot(kps_diff[i]));

        // judge large motion threld is 50 pixel
        if( dis > 50 * normal_unit_pixel)
        {
            inliner_set[i] = 0;
        }
        else
        {
            mean_pt_dis += dis;
            raw_inlier_cnt++;
        }

        // for debug 
        // LOG(INFO) << "the cur feature move is " << dis << endl;
    }

    mean_pt_dis /= raw_inlier_cnt;

    LOG(INFO) << "the mean pt dis is " << mean_pt_dis << "threlshold is " << normal_unit_pixel <<  " raw inliner cnt is "<<
             raw_inlier_cnt << endl;

    // when current inlier number less than 3, set all point to outlier

    if(raw_inlier_cnt < 3)
    {
        for(auto &inlier:inliner_set) {inlier = 0;}
        return ; 
    }


    // check if the motion is degenerated, if is, don't do ransac
    if(mean_pt_dis < normal_unit_pixel)
    {
        for(uint32_t i = 0; i < kps_diff.size(); i++)
        {
            if(inliner_set[i] == 0) continue;
            if(sqrt(kps_diff[i].dot(kps_diff[i])) > ransac_threld*normal_unit_pixel)
            {
                inliner_set[i] = 0;
            }
        }
        return;
    }

    // do ransac
    std::vector<int> raw_index(raw_inlier_cnt);
    for (int i = 0 ; i < raw_inlier_cnt; i++)
    {
        raw_index[i] = i;
    }

    LOG(INFO) << "iter max time is " << iter_max << endl;

    std::vector<int> best_inlier_opti(0);
    double best_error = 0;
    for(uint32_t iter_cnt = 0; iter_cnt < iter_max; iter_cnt++)
    {
        std::vector<int> ret_index;
        randperm(raw_index, 2, ret_index);

        int select_index1 = raw_index[ret_index[0]];
        int select_index2 = raw_index[ret_index[1]];

        Eigen::Vector2d diff_tx(diff_set(select_index1, 0), diff_set(select_index2, 0));
        Eigen::Vector2d diff_ty(diff_set(select_index1, 1), diff_set(select_index2, 1));
        Eigen::Vector2d diff_tz(diff_set(select_index1, 2), diff_set(select_index2, 2));

        // lpNorm<p> mean calculate the p norm of the matrix, in following code, p = 1,
        // mean calculate the 1 norm of matrix, is the sum of all matrix element's absoulute value 
        vector<double> norm_diff(3);
        norm_diff[0] = diff_tx.lpNorm<1>();
        norm_diff[1] = diff_ty.lpNorm<1>();
        norm_diff[2] = diff_tz.lpNorm<1>();

        int t_select = min_element(norm_diff.begin(), norm_diff.end()) - norm_diff.begin();

        Eigen::Vector3d model;

        if(t_select == 0)
        {
            Eigen::Matrix2d A;
            A << diff_ty, diff_tz;
            Eigen::Vector2d tyz = A.inverse()*(-diff_tx);
            model(0) = 1;
            model(1) = tyz(0);
            model(2) = tyz(1);
        }
        else if (t_select == 1)
        {
            Eigen::Matrix2d A;
            A << diff_tx, diff_tz;
            Eigen::Vector2d txz = A.inverse()*(-diff_ty);
            model(0) = txz(0);
            model(1) = 1;
            model(2) = txz(1);
        }
        else if (t_select == 2)
        {
            Eigen::Matrix2d A;
            A << diff_tx, diff_ty;
            Eigen::Vector2d txy = A.inverse()*(-diff_tz);
            model(0) = txy(0);
            model(1) = txy(1);
            model(2) = 1;
        }

        Eigen::VectorXd error = diff_set*model;

        vector<int> inlier_opti;

        for(uint32_t i = 0; i < error.rows(); i++)
        {
            if(!inliner_set[i])
            {
                continue;   
            }

            if(std::abs(error(i)) < ransac_threld*normal_unit_pixel)
            {
                inlier_opti.push_back(i);
            }
            // for debug
            // LOG(INFO) << "current error is " << error(i) << " error threld is "<< ransac_threld*normal_unit_pixel<<  endl;
        }

        // for debug
        // LOG(INFO) << "inliner size is " << inlier_opti.size();
        // if the  inliers number is small, do next ransac
        if (inlier_opti.size() < 0.2 * inliner_set.size())
        {
            LOG(INFO) << "inliner set is too little " << inlier_opti.size() << ' ' << inliner_set.size() << endl;
            continue;
        }

        // use inlier_opti estimate the translate
        Eigen::VectorXd batch_diff_tx(inlier_opti.size());
        Eigen::VectorXd batch_diff_ty(inlier_opti.size());
        Eigen::VectorXd batch_diff_tz(inlier_opti.size());

        for(uint32_t i = 0; i < inlier_opti.size(); i++)
        {
            batch_diff_tx(i) = diff_set(inlier_opti[i], 0);
            batch_diff_ty(i) = diff_set(inlier_opti[i], 1);
            batch_diff_tz(i) = diff_set(inlier_opti[i], 2);
        }

        if(t_select == 0)
        {
            Eigen::MatrixXd A(inlier_opti.size(), 2);
            A << batch_diff_ty, batch_diff_tz;
            Eigen::Vector2d tyz;
            tyz = (A.transpose()*A).inverse()*A.transpose()*(-batch_diff_tx);
            model(0) = 1;
            model(1) = tyz(0);
            model(2) = tyz(1);
        }
        else if(t_select == 1)
        {
            Eigen::MatrixXd A(inlier_opti.size(), 2);
            A << batch_diff_tx, batch_diff_tz;
            Eigen::Vector2d txz;
            txz = (A.transpose()*A).inverse()*A.transpose()*(-batch_diff_ty);
            model(0) = txz(0);
            model(1) = 1;
            model(2) = txz(1);
        }
        else if (t_select == 2)
        {
            Eigen::MatrixXd A(inlier_opti.size(), 2);
            A << batch_diff_tx, batch_diff_ty;
            Eigen::Vector2d txy;
            txy = (A.transpose()*A).inverse()*A.transpose()*(-batch_diff_tz);
            model(0) = txy(0);
            model(1) = txy(1);
            model(2) = 1;
        }

        Eigen::VectorXd new_error = diff_set*model;
        double opti_error = 0;

        for(uint32_t i = 0; i < inlier_opti.size(); i++)
        {
            if(inliner_set[inlier_opti[i]])
            {
                opti_error += fabs(new_error(inlier_opti[i]));
            }
        }

        opti_error /= inlier_opti.size();
        // LOG(INFO) << "inliner error after opti use inliner set is " << opti_error << endl;

        if (inlier_opti.size() > best_inlier_opti.size())
        {
            best_error = opti_error;
            best_inlier_opti = inlier_opti;
        }
    }

    inliner_set.clear();
    inliner_set.resize(kps1.size(), 0);
    for(uint32_t i = 0; i < best_inlier_opti.size(); i++)
    {
        inliner_set[best_inlier_opti[i]] = 1;
    }

}

template <typename T>
void ImgProcessor::RemoveOutlier(std::vector<T> &input_v, 
                            std::vector<uchar> &inliner, std::vector<T> &output_v)
{
    if(input_v.size() != inliner.size())
    {
        LOG(INFO) << "input size not equal to inliner size " << endl;
        return;
    }
    output_v.clear();
    for(uint32_t i = 0; i < input_v.size(); i++)
    {
        if(inliner[i])
        {
            output_v.push_back(input_v[i]);
        }
    }
}

void ImgProcessor::rescalePoints(std::vector<cv::Point2f> &kps1, 
                    std::vector<cv::Point2f> &kps2, float &scaling_factor)
{
    scaling_factor = 0.0f;

    for (uint32_t i = 0; i < kps1.size(); i++)
    {
        scaling_factor += sqrt(kps1[i].dot(kps1[i]));
        scaling_factor += sqrt(kps2[i].dot(kps2[i]));
    }

    scaling_factor = (kps1.size() + kps2.size())/scaling_factor*sqrt(2.0);

    for (uint32_t i = 0; i < kps1.size(); i++)
    {
        kps1[i] *= scaling_factor;
        kps2[i] *= scaling_factor;
    }
}

void ImgProcessor::setMsckf_Filter(std::shared_ptr<OC_MSCKF_Filter> ptr)
{
    filter_ptr_ = ptr;
}
