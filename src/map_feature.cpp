#include "map_feature.hpp"
#include <memory>
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include "cam_state.hpp"

using namespace vins_slam;
using namespace std;

//TODO:code review
void MapFeature::add_observation(unsigned long int cam_id, obs_data_t &obs)
{
    std::unique_lock<std::mutex> lk(obs_data_mutex);
    if(obs_map.find(cam_id) != obs_map.end())
    {
        LOG(INFO) << "obs have add to camera " << cam_id << endl;
    }
    else
    {
        obs_map.insert(std::make_pair(cam_id, obs));
    }
}

void MapFeature::remove_observation(unsigned long int cam_id)
{
    std::unique_lock<std::mutex> lk(obs_data_mutex);

    if(obs_map.find(cam_id) == obs_map.end())
    {
        LOG(INFO) << "obs is not in camera " << cam_id << endl;
    }
    else
    {   
        obs_map.erase(cam_id);
    }   
}

// TODO:check initial guess and jacobian and cost calculate function

// TODO: check the initial observe error befor opti  in  stereo msckf

// TODO: check the first and last pose trans of observed cam states,
// for direct think, the left and right camera can get a enough motion
// to init mappoint?
bool MapFeature::initial_position(CamStateSever &cam_states, Eigen::Matrix4d &T_rcam_lcam)
{
    // change the camera pose from world frame to first frame, 
    // mean while add right camera pose

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> cam_poses;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> cam_measures;

    // LOG(INFO) << "T_lcam_rcam is" << endl << T_rcam_lcam.inverse() << endl;
    for(auto obs : obs_map)
    {
        if(cam_states.find(obs.first) != cam_states.end())
        {
            Eigen::Matrix3d cam_rotate = cam_states[obs.first].rotation_matrix.transpose();
            Eigen::Matrix4d cam_pose;
            cam_pose.setIdentity();
            cam_pose.block<3,1>(0,3) = cam_states[obs.first].position;
            cam_pose.block<3,3>(0,0) = cam_rotate;

            Eigen::Matrix4d rcam_pose;
            // rcam_pose = T_rcam_lcam*cam_pose;
            rcam_pose = cam_pose*T_rcam_lcam.inverse();

            cam_poses.push_back(cam_pose);
            cam_poses.push_back(rcam_pose);

            Eigen::Vector2d lcam_obs(obs.second.un_u0, obs.second.un_v0);
            Eigen::Vector2d rcam_obs(obs.second.un_u1, obs.second.un_v1);
            cam_measures.push_back(lcam_obs);
            cam_measures.push_back(rcam_obs);
        }
        else
        {
            LOG(INFO) << "cur camera state id is not exist " << endl;
        }
    }

    // get the transform from oldest camera frame to other camera frames observed the feature
    Eigen::Matrix4d T_w_c0 = cam_poses[0];
    for (auto &cam_pose:cam_poses)
    {
        cam_pose = cam_pose.inverse()*T_w_c0;
    }

    // use the last and oldest camera relavtive pose and observation to estimate a 
    // initial position
    // p2 = R*p1 + t ---> 1/z2*p2 = 1/z1*R*p1 + 1/z1*t, note this equal have a proportion factor,
    // use the obs of p2 and p1 can build a least square prob, could get a analytic solution

    // for debug
    // LOG(INFO) << "Tci_c0 is " << endl << cam_poses[cam_poses.size() - 1] << endl << " cam_measures c0 " << cam_measures[0].transpose()
    //           << " cam_measures ci " << cam_measures[cam_measures.size() - 1].transpose() << endl;

    Eigen::Vector3d Pc;
    initGuess(cam_poses[cam_poses.size() - 1], cam_measures[0], cam_measures[cam_measures.size() - 1], Pc);
    
    // use the normalized image plane coordinates and invert depth as opti param
    Eigen::Vector3d position(Pc(0)/Pc(2), Pc(1)/Pc(2), 1/Pc(2));

    // judge the depth is positive
    if(Pc(2) < 0)
    {
        LOG(INFO) << "init mappoint guess get a invalid negative depth, it's obs is :" << endl;
        for(auto obs:obs_map)
        {
            LOG(INFO) << "cam id " << obs.first << " obs img coordinate " << obs.second.u0 << ' ' << 
                    obs.second.v0 << ' ' << obs.second.u1 << ' ' << obs.second.v1 << endl;
        }
        return false;
    }

    // use the initial position do LM nonlinear optimization, 
    // note that lamda change time in one optimize have a threlshold

    // calculate the observe error when use initial position
    double this_error = 0;
    for(uint32_t i = 0; i < cam_measures.size(); i++)
    {
        double this_cost = 0;
        cost(cam_poses[i], position, cam_measures[i], this_cost);
        this_error += this_cost;
    }

    // for debug
    // LOG(INFO) << "the  square error befor position opt is " << this_error << endl;
    // LOG(INFO) << "the error befor postion opt is " << sqrt(this_error) << endl;
    
    int iter_cnt = 0;
    double total_error = 0;
    double lamda = opti_config.initial_damping;

    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    Eigen::Vector3d delta_x;

    do
    {
        A.setZero();
        b.setZero();
        // calculate jacobain for every measurement, and accmulate the Hessian and b
        for(uint32_t i = 0; i < cam_measures.size(); i++)
        {
            Eigen::Matrix<double,2,3> J_cur;
            Eigen::Vector2d r;
            double w;
            jacobaincalc(cam_poses[i], position, cam_measures[i], J_cur, r, w);

            if(w <= 1)
            {
                A += J_cur.transpose()*J_cur;
                b += J_cur.transpose()*r;
            }
            else
            {
                double w_2 = w*w;
                A += J_cur.transpose()*J_cur*w_2;
                b += J_cur.transpose()*r*w_2;
            }
        }


        Eigen::Vector3d update_position;
        int inner_iter_cnt = 0;
        bool is_cost_reduce = false;

        do
        {
            Eigen::Matrix3d A_damp = A+lamda*Eigen::Matrix3d::Identity();
            delta_x = A_damp.ldlt().solve(b);
            Eigen::Vector3d update_position = position - delta_x; // add or reduce depend on the jacobain

            // for debug
            // LOG(INFO) << "update position is " << update_position.transpose() <<
            //                 " delta x is " << delta_x << " delta x norm is " 
            //                 << delta_x.norm() << endl;
            for(uint32_t i = 0; i < cam_measures.size(); i++)
            {
                double this_cost = 0;
                cost(cam_poses[i], update_position, cam_measures[i], this_cost);
                total_error += this_cost;
            }

            if(total_error > this_error)
            {
                is_cost_reduce = false;
                lamda = (lamda*10 < 1e7)? lamda*10 : 1e7; 
            }
            else
            {
                is_cost_reduce = true;
                position = update_position;
                this_error = total_error;
                lamda = (lamda/10 > 1e-12)? lamda/10:1e-12;
            }

            // for debug

            // LOG(INFO) << "this inlier loop error is " << this_error << " last error is " << total_error << endl;

            inner_iter_cnt++;

        } while (inner_iter_cnt < opti_config.inner_loop_max_iteration &&
                  !is_cost_reduce);
        
        // for debug
        // LOG(INFO) << "this iter final error is " << this_error << endl;
        
        iter_cnt++;
    } while(iter_cnt <= opti_config.outer_loop_max_iteration &&
            delta_x.norm() > opti_config.estimation_precision);
    
    
    Eigen::Vector3d final_position(position(0)/position(2), position(1)/position(2), 
                    1/position(2));


    // change the coordinate to world frame, 
    // meanwhile transform to all cameras frames observe it,
    // if one depth is negative, the position initial failed
    for(auto cam_pose:cam_poses)
    {
        Eigen::Vector3d Pc = cam_pose.block<3,3>(0,0)*final_position + 
                                cam_pose.block<3,1>(0,3);
        if(Pc(2) < 0)
        {
            LOG(INFO) << "cur depth is negative " << endl;
            is_initial = false;
            return false;
        }
    }

    // for debug
    // LOG(INFO) << "final position is " << final_position.transpose() << endl;

    pw = T_w_c0.block<3,3>(0,0)*final_position + T_w_c0.block<3,1>(0,3);
    
    is_initial = true;
    return true;

}

void MapFeature::initGuess(Eigen::Matrix4d &T_ci_c0, Eigen::Vector2d &z0,
                        Eigen::Vector2d &zi, Eigen::Vector3d &Pc)
{
    
    Eigen::Vector3d p1(z0(0),z0(1),1.0);

    Eigen::Vector3d m = T_ci_c0.block<3,3>(0,0) * p1;

    Eigen::Vector3d p2(zi(0), zi(1), 1);

    Eigen::Vector2d A;
    A(0) = p2(0)*m(2) - m(0);
    A(1) = p2(1)*m(2) - m(1);

    Eigen::Vector3d translation = T_ci_c0.block<3,1>(0,3);

    Eigen::Vector2d b;
    b(0) = translation(0) - p2(0)*translation(2);
    b(1) = translation(1) - p2(1)*translation(2);

    double depth;
    depth = (A.transpose()*A).inverse()*A.transpose()*b;

    Pc = p1*depth;

    // for debug
    // LOG(INFO) << "the Pc pos is " << Pc.transpose() << " depth is " << depth << endl;
    

    // use triangulate get the initial postion

    /*
    Eigen::Matrix4d T0 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d A;
    A.row(0) = T0.row(1) - z0(1)*T0.row(2);
    A.row(1) = T0.row(0) - z0(0)*T0.row(2);
    A.row(2) = T_ci_c0.row(1) - zi(1)*T_ci_c0.row(2);
    A.row(3) = T_ci_c0.row(0) - zi(0)*T_ci_c0.row(2);

    Eigen::BDCSVD<Eigen::Matrix4d> svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d P = svd.matrixV().col(3);
    P /= P(3);

    if(P(2) < 0)
    {
        LOG(WARNING) << "postion depth negative " << endl;
    }
    LOG(INFO) << "cur Pc position is " << P.segment<3>(0).transpose() << endl;

    Pc = P.segment<3>(0);
    */
}

bool MapFeature::check_motion(CamStateSever &cam_states)
{
    // use the last and oldest obs camera pose to caculate a relative tanslation,
    // and change it to a parrellal translation by feature direction, 
    // use parrellal translation to judge the motion is have enough parallex

    // for debug, check the first and last obs is in cam_states
    auto first_iter = obs_map.begin();
    auto last_iter = --obs_map.end();



    if(cam_states.find(first_iter->first) == cam_states.end() 
        || cam_states.find(last_iter->first) == cam_states.end())
    {
        LOG(INFO) << "first or last iter is not in cam states, this should not happen" << endl;
    }

    Eigen::Vector3d last_cam_trans = cam_states[last_iter->first].position;

    Eigen::Vector3d oldest_cam_trans = cam_states[first_iter->first].position;
    Eigen::Matrix3d oldest_cam_rotation = cam_states[first_iter->first].rotation_matrix;

    Eigen::Vector3d relative_trans = last_cam_trans - oldest_cam_trans;

    Eigen::Vector3d feature_dir;
    feature_dir(0) = first_iter->second.un_u0;
    feature_dir(1) = first_iter->second.un_v0;
    feature_dir(2) = 1;
    feature_dir /= feature_dir.norm();

    // change the feature direction to world frame 
    Eigen::Vector3d feature_dir_w = oldest_cam_rotation.transpose()*feature_dir;

    double len = relative_trans.transpose()*feature_dir_w;

    Eigen::Vector3d orthogonal_trans = relative_trans - len*feature_dir_w;

    // PROB: orthogonal_trans have numeric problem
    // I1104 11:45:28.907898  6746 map_feature.cpp:254] cur feature orthogonal is -nan -nan -nannorm is -nan
    // LOG(INFO) << "cur feature orthogonal is " << orthogonal_trans.transpose()
    //           << "norm is " << orthogonal_trans.norm() << endl;

    if(orthogonal_trans.norm() > opti_config.translation_threlshold)
    {
        return true;
    }
    else
    {
        return false;
    }

}

void MapFeature::jacobaincalc(Eigen::Matrix4d &T, Eigen::Vector3d &Pc, Eigen::Vector2d &z, 
                                Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r, double &w)
{
    // note that Pc is composed by normalize imag plane coordinate and invert depth 
    Eigen::Vector3d Pr = T.block<3,3>(0,0)*Eigen::Vector3d(Pc(0), Pc(1), 1) + 
                         Pc(2)*T.block<3,1>(0,3);

    double z_2 = Pr(2)*Pr(2);
    Eigen::Matrix<double,2,3> J_z_Pr;

    J_z_Pr << 1/Pr(2),0, -Pr(0)/z_2, 0, 1/Pr(2), -Pr(1)/z_2;

    Eigen::Matrix<double,3,3> J_Pr_Pc;
    // J_Pr_Pc = T.block<3,3>(0,0);
    J_Pr_Pc.leftCols(2) = T.block<3,2>(0,0);
    J_Pr_Pc.rightCols(1) = T.block<3,1>(0,3);

    // calculate the jacobain by chain rule
    J = J_z_Pr*J_Pr_Pc;

    r(0) = Pr(0)/Pr(2) - z(0);
    r(1) = Pr(1)/Pr(2) - z(1);

    // use huber cost func get the error weight,
    // when error little than huber epsilon, cost function is error^2,
    // when error greater than huber epsilon, cost function is 2*huber_epislon*error - error^2,
    // in following code, omit the quadratic term, so weight is calculate by following formular:
    // w = sqrt(error^2)/error when error is little than huber_epsilon,
    // w = sqrt(2*huber_epsilon*error)/error = sqrt(2*huber_epsilon/error) when
    // error is greater than huber_epsilon

    if(r.norm() < opti_config.huber_epsilon)
    {
        w = 1;
    }
    else
    {
        w = sqrt(2*opti_config.huber_epsilon/r.norm());
    }
}

void MapFeature::cost(Eigen::Matrix4d &T, Eigen::Vector3d &P, 
        Eigen::Vector2d &z, double &e)
{
    // Eigen::Vector3d Pc = T.block<3,3>(0,0)*P + T.block<3,1>(0,3);
    Eigen::Vector3d Pc = T.block<3,3>(0,0)*Eigen::Vector3d(P(0), P(1), 1.0) + 
                            P(2)*T.block<3,1>(0,3);

    Eigen::Vector2d r;
    r(0) = Pc(0)/Pc(2) - z(0);
    r(1) = Pc(1)/Pc(2) - z(1);

    e = r.squaredNorm();
}