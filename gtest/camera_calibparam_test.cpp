#include <gtest/gtest.h>
#include <glog/logging.h>
#include <eigen3/Eigen/Core>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
#include <memory>
#include <string>
#include "camera.hpp"
#include "frame.hpp"

using namespace testing;
using namespace vins_slam;
using namespace cv;

int main(int argc, char** argv)
{
    InitGoogleTest(&argc, argv);

    RUN_ALL_TESTS();
}


string left_img_path = "../gtest/img_data/cam0";
string right_img_path = "../gtest/img_data/cam1";

// test the camera undisort and disort model
TEST(Camera_Calibparam_test, undisort_test)
{
    cv::Mat left_img;
    cv::Mat right_img;

    // create detector
    int max_feature = 200;
    double minDistance = 20;
    int blocksize = 3;
    Ptr<GFTTDetector> detector = GFTTDetector::create(max_feature, 0.01, 
                                        minDistance, blocksize, true);

    left_img = cv::imread(left_img_path+"/2.png", cv::IMREAD_GRAYSCALE);
    right_img = cv::imread(right_img_path+"/2.png", cv::IMREAD_GRAYSCALE);

    Eigen::Matrix<double,3,3> left_K_;

    left_K_ << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1;

    Eigen::Matrix<double,3,3> right_K_;

    right_K_ << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0 ,1;

    Eigen::Matrix4d left_cam_ext, right_cam_ext;
    left_cam_ext <<  0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0;
    right_cam_ext << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                    0.0, 0.0, 0.0, 1.0;

    std::shared_ptr<Camera> left_cam(new Camera(left_K_, left_cam_ext, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05));
    

    std::shared_ptr<Camera> right_cam(new Camera(right_K_, right_cam_ext, -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05));

    std::shared_ptr<Frame> cur_frame_(new Frame(left_img, right_img, 0));

    std::vector<KeyPoint> features;
    detector->detect(left_img, features);

    double left_sum_error = 0;
    int valid_cnt = 0;
    for(auto kp : features)
    {
        Eigen::Vector3d left_img_un_coor;
        Eigen::Vector2d left_img_dis(kp.pt.x, kp.pt.y);
        left_cam->unproject(left_img_dis, left_img_un_coor);

        Eigen::Vector2d reproject_img_dis;
        left_cam->project(left_img_un_coor, reproject_img_dis);
        left_sum_error += (left_img_dis-reproject_img_dis).norm();
        valid_cnt++;
    }

    LOG(INFO) << "the left mean sqrt square error is " << left_sum_error/valid_cnt;

}