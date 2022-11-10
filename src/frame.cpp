#include "frame.hpp"
#include "opencv/cv.hpp"
#include <glog/logging.h>

namespace vins_slam {

using namespace cv;
using namespace std;

// in frame construct, only do left_img and right_img copy and store timestamp and detector,
// other operation wraped in other class member function
// 1.Class frame have a function to track the last frame and store the tracked feature
// 2.Class frame have a function to detect the new feature and store the tracked extract feature
// 3.Befor add a frame to MSCKF_Filter, finish the 1,2
Frame::Frame(cv::Mat left_img, cv::Mat right_img, double timestamp)
{
    left_img.copyTo(left_img_);
    right_img.copyTo(right_img_);
    timestamp_ = timestamp;
    // LOG(INFO) << "timestamp is " << timestamp_ << endl;
}
}