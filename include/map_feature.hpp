#pragma once
#include <eigen3/Eigen/Core>
#include <map>
#include <mutex>
#include "cam_state.hpp"

namespace vins_slam {

// class CamStateSever;

typedef struct obs_data {
    double u0;
    double v0;
    double un_u0;
    double un_v0;
    double u1;
    double v1;
    double un_u1;
    double un_v1;
} obs_data_t;

struct OptimizationConfig {
    double translation_threlshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    double outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
        // translation_threlshold(0.2),
        translation_threlshold(-1.0),
        huber_epsilon(0.01),
        estimation_precision(5e-7),
        initial_damping(1e-3),
        outer_loop_max_iteration(10),
        inner_loop_max_iteration(10) {
            return;
        }
};

class MapFeature {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapFeature(int id) {feature_id = id;}
    MapFeature() {}
    ~MapFeature() {}

    // MapFeature& operator=(MapFeature &other)
    // {
    //     MapFeature mappoint(other.feature_id);
    //     return *this;
    // }
    int feature_id;
    void add_observation(unsigned long int cam_id, obs_data_t &obs);
    void remove_observation(unsigned long int cam_id);

    // decide whether do a position init by judge the motion of the current feature
    bool check_motion(CamStateSever &cam_states);
    // do a position init by multi observations
    bool initial_position(CamStateSever &cam_states, Eigen::Matrix4d &T_rcam_lcam);

    // do a init guess by two measurement and relative cam pose
    void initGuess(Eigen::Matrix4d &T_ci_c0, Eigen::Vector2d &z0,
                         Eigen::Vector2d &zi, Eigen::Vector3d &Pc);

    // caculate a jacobain for a camera meansurement
    void jacobaincalc(Eigen::Matrix4d &T, Eigen::Vector3d &Pc, Eigen::Vector2d &z, 
                                Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r, double &w);

    // caculate cost of estimate mappoint position
    void cost(Eigen::Matrix4d &T, Eigen::Vector3d &P, 
            Eigen::Vector2d &z, double &e);

    std::map<unsigned long int, obs_data_t, std::less<unsigned long int>> obs_map;
    Eigen::Vector3d pw;
    bool is_initial = false;
    std::mutex obs_data_mutex;

    struct OptimizationConfig opti_config;

};

}