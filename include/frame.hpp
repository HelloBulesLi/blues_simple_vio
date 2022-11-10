#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include "camera.hpp"

namespace vins_slam {

using namespace std;
using namespace cv;

class Frame {
public:
    Frame(cv::Mat left_img, cv::Mat right_img, double timestamp);
    double timestamp_;
    cv::Mat left_img_;
    cv::Mat right_img_;
};

}