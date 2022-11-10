#include "imu_measure.hpp"
#include <memory>
#include <glog/logging.h>

using namespace vins_slam;
using namespace std;

int global_imu_measure_id = 0;

IMU_Measure::IMU_Measure(Eigen::Vector3d &ang_vel, Eigen::Vector3d &acc_vel, double timestamp)
{
    angular_vel = ang_vel;
    accelarator = acc_vel;
    timestamp_ = timestamp;
    imu_measure_id_ =  global_imu_measure_id;
    global_imu_measure_id++;
    // LOG(INFO) << "cur imu timestamp is " << timestamp_ << endl;
}

