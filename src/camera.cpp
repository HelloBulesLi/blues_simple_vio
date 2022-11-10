#include "camera.hpp"
#include <glog/logging.h>

using namespace vins_slam;


// void project(Eigen::Vector3d &pc, Eigen::Vector2d img_pos);
// void unproject(Eigen::Vector2d &img_pos, Eigen::Vector3d &gray_dir);

void Camera::project(Eigen::Vector3d &pc, Eigen::Vector2d &img_pos)
{
    Eigen::Vector3d norm_plan_pos = pc/pc(2);

    double r = norm_plan_pos(0)*norm_plan_pos(0) + norm_plan_pos(1)*norm_plan_pos(1);
    double corr = norm_plan_pos(0)*norm_plan_pos(1);
    
    double u_dis = norm_plan_pos(0)*(1 + k1_*r + k2_*r*r) + 2*p1_*corr + p2_*(r+2* norm_plan_pos(0)* norm_plan_pos(0));
    double v_dis = norm_plan_pos(1)*(1 + k1_*r + k2_*r*r) + 2*p2_*corr + p1_*(r+2* norm_plan_pos(1)* norm_plan_pos(1));

    img_pos(0) = u_dis*fx_ + cx_;
    img_pos(1) = v_dis*fy_ + cy_;
}

void Camera::projectun(Eigen::Vector3d &pc, Eigen::Vector2d img_norm_pos)
{
    // Eigen::Vector3d norm_plan_pos = pc/pc(2);

    // double r = norm_plan_pos(0)*norm_plan_pos(0) + norm_plan_pos(1)*norm_plan_pos(1);
    // double corr = norm_plan_pos(0)*norm_plan_pos(1);
}

void Camera::unproject(Eigen::Vector2d &img_pos, Eigen::Vector3d &gray_dir)
{
    double x_dis,y_dis;
    x_dis = (img_pos(0) - cx_)/fx_;
    y_dis = (img_pos(1) - cy_)/fy_;

    int n = 8;
    Eigen::Vector2d d_u;
    distortion(Eigen::Vector2d(x_dis, y_dis), d_u);
    // Approximate value
    double mx_u = x_dis - d_u(0);
    double my_u = y_dis - d_u(1);

    for (int i = 1; i < n; ++i)
    {
        distortion(Eigen::Vector2d(mx_u, my_u), d_u);
        mx_u = x_dis - d_u(0);
        my_u = y_dis - d_u(1);
    }

    // Apply inverse distortion model
    // proposed by Heikkila

    // double mx_d = x_dis;
    // double my_d = y_dis;
    // double mx2_d = mx_d*mx_d;
    // double my2_d = my_d*my_d;
    // double mxy_d = mx_d*my_d;
    // double rho2_d = mx2_d+my2_d;
    // double rho4_d = rho2_d*rho2_d;
    // double radDist_d = k1_*rho2_d+k2_*rho4_d;
    // double Dx_d = mx_d*radDist_d + p2_*(rho2_d+2*mx2_d) + 2*p1_*mxy_d;
    // double Dy_d = my_d*radDist_d + p1_*(rho2_d+2*my2_d) + 2*p2_*mxy_d;
    // double inv_denom_d = 1/(1+4*k1_*rho2_d+6*k2_*rho4_d+8*p1_*my_d+8*p2_*mx_d);

    // double mx_u = mx_d - inv_denom_d*Dx_d;
    // double my_u = my_d - inv_denom_d*Dy_d;

    // LOG(INFO) << "befor dis img coor is " << x_dis << ' ' << y_dis << "after dis img coor is"
    //           << mx_u << ' ' << my_u << endl;
    gray_dir << mx_u, my_u, 1.0;

}

void Camera::undisort(Eigen::Vector2d &img_pos, Eigen::Vector2d &undis_img_pos)
{
    Eigen::Vector3d gray_dir;
    unproject(img_pos, gray_dir);
    project(gray_dir, undis_img_pos);
}

void Camera::undisort_normlize(Eigen::Vector2d &img_pos, Eigen::Vector2d &undis_img_pos)
{
    double x_dis,y_dis;
    x_dis = (img_pos(0) - cx_)/fx_;
    y_dis = (img_pos(1) - cy_)/fy_;

    int n = 8;
    Eigen::Vector2d d_u;
    distortion(Eigen::Vector2d(x_dis, y_dis), d_u);
    // Approximate value
    double mx_u = x_dis - d_u(0);
    double my_u = y_dis - d_u(1);

    for (int i = 1; i < n; ++i)
    {
        distortion(Eigen::Vector2d(mx_u, my_u), d_u);
        mx_u = x_dis - d_u(0);
        my_u = y_dis - d_u(1);
    }

    // Apply inverse distortion model
    // proposed by Heikkila

    // double mx_d = x_dis;
    // double my_d = y_dis;
    // double mx2_d = mx_d*mx_d;
    // double my2_d = my_d*my_d;
    // double mxy_d = mx_d*my_d;
    // double rho2_d = mx2_d+my2_d;
    // double rho4_d = rho2_d*rho2_d;
    // double radDist_d = k1_*rho2_d+k2_*rho4_d;
    // double Dx_d = mx_d*radDist_d + p2_*(rho2_d+2*mx2_d) + 2*p1_*mxy_d;
    // double Dy_d = my_d*radDist_d + p1_*(rho2_d+2*my2_d) + 2*p2_*mxy_d;
    // double inv_denom_d = 1/(1+4*k1_*rho2_d+6*k2_*rho4_d+8*p1_*my_d+8*p2_*mx_d);

    // double mx_u = mx_d - inv_denom_d*Dx_d;
    // double my_u = my_d - inv_denom_d*Dy_d;

    // LOG(INFO) << "befor dis img coor is " << x_dis << ' ' << y_dis << "after dis img coor is"
    //           << mx_u << ' ' << my_u << endl;
    undis_img_pos << mx_u, my_u;
}

void Camera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u)
{
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1_ * rho2_u + k2_ * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1_ * mxy_u + p2_ * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2_ * mxy_u + p1_ * (rho2_u + 2.0 * my2_u);
}

void Camera::project_undisort(Eigen::Vector3d &pc, Eigen::Vector2d &img_pos)
{
    Eigen::Vector3d norm_plan_pos = pc/pc(2);

    img_pos(0) = norm_plan_pos(0)*fx_ + cx_;
    img_pos(1) = norm_plan_pos(1)*fy_ + cy_;
}
