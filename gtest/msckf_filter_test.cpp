#include <gtest/gtest.h>
#include "msckf_filter.hpp"
#include "img_processor.hpp"
#include <fstream>
#include <string>
#include <thread>
#include <glog/logging.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/gl.hpp>
#include <pangolin/pangolin.h>

using namespace std;
using namespace testing;
using namespace vins_slam;

void img_process(std::shared_ptr<ImgProcessor> img_proc_ptr);
void img_read_thread(string &img_path, std::shared_ptr<ImgProcessor> img_proc_ptr);
void imu_read_thread(string &imu_path, std::shared_ptr<ImgProcessor> img_proc_ptr, 
                    std::shared_ptr<OC_MSCKF_Filter> filter_ptr);
void msckf_filter_process(std::shared_ptr<OC_MSCKF_Filter> filter_ptr);
void Draw_Traj(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &poses);

string data_path = "/media/libo/KINGSTON/Euroc_DataSet/V1_01_easy/mav0/";
string imu_path = data_path+"imu0/";

bool is_first_imu = true;

int main(int argc, char** argv)
{
    FLAGS_log_dir = "/home/libo/vins_msckf/build";
    // FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging("msckf_filter_test");
    InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    return 0;
}

TEST(msckf_filter, do_filter_test)
{
    std::shared_ptr<ImgProcessor> img_proc(new ImgProcessor());
    std::shared_ptr<OC_MSCKF_Filter> filter_ptr(new OC_MSCKF_Filter());

    img_proc->setMsckf_Filter(filter_ptr);

    std::thread img_proc_handle(img_process, img_proc);
    std::thread img_read_thread_handle(img_read_thread, std::ref(data_path), img_proc);
    std::thread imu_read_thread_handle(imu_read_thread, std::ref(imu_path),img_proc,filter_ptr);
    std::thread msckf_filter_thread_handle(msckf_filter_process, filter_ptr);
    

    img_proc_handle.join();
    img_read_thread_handle.join();
    imu_read_thread_handle.join();
    msckf_filter_thread_handle.join();
}

void msckf_filter_process(std::shared_ptr<OC_MSCKF_Filter> filter_ptr)
{
    while(true)
    {
        filter_ptr->process();
    }
}

void img_process(std::shared_ptr<ImgProcessor> img_proc_ptr)
{
    while(true)
    {
        img_proc_ptr->process();
    }
}

void img_read_thread(string &img_path, std::shared_ptr<ImgProcessor> img_proc_ptr)
{
    int maxCorner = 200;
    double minDistance = 30;
    int blocksize = 3;
    Ptr<GFTTDetector> detector = GFTTDetector::create(200, 0.01, minDistance, blocksize, true);

    string left_img_path = img_path+"cam0/";
    string right_img_path = img_path+"cam1/";
    ifstream left_img_data;
    ifstream right_img_data;
    
    left_img_data.open(left_img_path+"data.csv", ios::in);
    
    if(!left_img_data.is_open())
    {
        LOG(ERROR) << "can't open left img data csv " << endl;
        return; 
    }

    right_img_data.open(right_img_path+"data.csv", ios::in);

    if(!right_img_data.is_open())
    {
        LOG(ERROR) << "can't open right img data csv " << endl;
    }

    double first_img_timestamp = 1403715273262140000;
    first_img_timestamp /= 1e9;
    first_img_timestamp = first_img_timestamp + 0;

    double left_timestamp = 0,right_timestamp = 0;
    string left_img_name,right_img_name;

    string title;
    char dot;
    left_img_data >> title >> title;
    right_img_data >> title >> title;


    while(left_timestamp < first_img_timestamp)
    {
        left_img_data >> left_timestamp >> dot >> left_img_name;
        right_img_data >> right_timestamp >> dot >> right_img_name;
        left_timestamp /= 1e9;
        right_timestamp /= 1e9;
    }

    LOG(INFO) << "first img measure ment timestamp is " << std::setprecision(15) << left_timestamp << endl;

    
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

    std::shared_ptr<Camera> left_cam(new Camera(left_K_, left_cam_ext, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05));
    std::shared_ptr<Camera> right_cam(new Camera(right_K_, right_cam_ext, -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05));

    int img_cnt = 2912;
    // int img_cnt = 400;
    int count = 0;
    bool first_img = true;
    std::shared_ptr<Frame> pre_frame;

    while(count < img_cnt)
    {
        if(is_first_imu)
        {
            continue;
        }

        auto t1 = std::chrono::steady_clock::now(); 
        cv::Mat left_img,right_img;
        left_img = cv::imread(left_img_path+"data/"+left_img_name, cv::IMREAD_GRAYSCALE);
        right_img = cv::imread(right_img_path+"data/"+right_img_name, cv::IMREAD_GRAYSCALE);
        
        std::shared_ptr<Frame> cur_frame(new Frame(left_img, right_img, left_timestamp));

        img_proc_ptr->addframe(cur_frame);


        left_img_data >> left_timestamp >> dot >> left_img_name;
        right_img_data >> right_timestamp >> dot >> right_img_name;
        left_timestamp /= 1e9;
        right_timestamp /= 1e9;


        count++;
        auto t2 = std::chrono::steady_clock::now(); 
        double cost_time = std::chrono::duration<double, std::milli>(t2-t1).count();
        
        if(cost_time < 50)
        {
            usleep(50*1000 - cost_time*1000);
        }
        auto t3 = std::chrono::steady_clock::now();

        // for debug
        // LOG(INFO) << "add img interval time is " <<  
        //             std::chrono::duration<double, std::milli>(t3-t1).count() << " ms";
    }

    LOG(INFO) << "pub img cnt is " << count << endl;

    return ;
}

void imu_read_thread(string &imu_path, std::shared_ptr<ImgProcessor> img_proc_ptr, 
                    std::shared_ptr<OC_MSCKF_Filter> filter_ptr)
{
    ifstream imu_data;
    imu_data.open(imu_path+"data.csv", ios::in);

    if(!imu_data.is_open())
    {
        LOG(ERROR) << " imu data csv can't open " << endl;
    }

    double first_imu_timestamp = 1403715273262140000;
    first_imu_timestamp /= 1e9;
    first_imu_timestamp = first_imu_timestamp + 0;

    string first_line;
    for(int i = 0; i < 14; i++)
    {
        imu_data >> first_line;
        // LOG(INFO) << first_line;
    }

    double imu_timestamp = 0;
    Eigen::Vector3d angle_vel;
    Eigen::Vector3d acc_vel;
    char dot;

    while(imu_timestamp < first_imu_timestamp)
    {
        imu_data >> imu_timestamp >> dot >> angle_vel(0) >> dot 
                >> angle_vel(1) >> dot >> angle_vel(2) >> dot 
                >> acc_vel(0) >> dot >> acc_vel(1) >> dot >> acc_vel(2);
        imu_timestamp /= 1e9;
    }

    LOG(INFO) << "first imu measure ment timestamp is " << std::setprecision(15) << imu_timestamp << endl;
   
    int imu_cnt = 29120;
    // int imu_cnt = 22*200;
    int count = 0;

    while(count < imu_cnt)
    {
        if(is_first_imu)
        {
            is_first_imu = false;
        }

        auto t1 = std::chrono::steady_clock::now();
        std::shared_ptr<IMU_Measure> imu(new IMU_Measure(angle_vel, acc_vel, imu_timestamp));
        img_proc_ptr->addimu(imu);
        filter_ptr->AddImu(imu);

        imu_data >> imu_timestamp >> dot >> angle_vel(0) >> dot 
                >> angle_vel(1) >> dot >> angle_vel(2) >> dot 
                >> acc_vel(0) >> dot >> acc_vel(1) >> dot >> acc_vel(2);
        imu_timestamp /= 1e9;
        auto t2 = std::chrono::steady_clock::now();

        double cost_time = std::chrono::duration<double, std::milli>(t2-t1).count();
        if(cost_time < 5)
        {
            usleep(5000 - cost_time*1000);
        }
        auto t3 = std::chrono::steady_clock::now();

        count++;
        //for debug
        // LOG(INFO) << "imu add interval time is " << 
        //             std::chrono::duration<double, std::milli>(t3-t1).count() << " ms"<< endl;
    }

    LOG(INFO) << "pub imu cnt is " << count << endl;
    
    return ;
}


void Draw_Traj(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &poses)
{
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);

    const int gl_camera_width = 752;
    const int gl_camera_height = 480;
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;

    
    glEnable(GL_DEPTH_TEST); // only show the point in front
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Eigen::Matrix4d test;
    test << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
            0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4f test1 = test.template cast<float>();
    Eigen::Matrix4f test2 = test1.inverse();

    pangolin::OpenGlMatrix m1(test1);

    pangolin::OpenGlRenderState vis_cam(
    pangolin::ProjectionMatrix(
        gl_camera_width, gl_camera_height,
        fx,fy,cx,cy,0.1,1000),
        // pangolin::ModelViewLookAt(0, -5, 10, 0, 0, 0, 0, -1, 0)
        // pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0)*m1
        // pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0)
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0, -1, 0)
        );

 // auto &view = pangolin::CreateDisplay().SetAspect((float)gl_camera_width/(float)gl_camera_height);
    auto &view = pangolin::CreateDisplay().SetBounds(0,1.0,0,1.0, -1024.0f/768.0f)
                                          .SetHandler(new pangolin::Handler3D(vis_cam));
    
    bool first_print = true;
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        view.Activate(vis_cam);
        // glClearColor(1.0f,1.0f,1.0f,1.0f); // white backgroud color ?
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw pose
        float sz = 0.1;
        const float width = 752;
        const float height = 480;

        // Eigen::Matrix4d Twc = poses[0].inverse();
        Eigen::Matrix4d Twc = poses[0];
        Eigen::Vector3d pre_pos = Twc.block<3,1>(0,3);
    
        glPointSize(20);
        glBegin(GL_POINTS);
        glColor3f(0, 1, 0);
        glVertex3d(pre_pos(0), pre_pos(1), pre_pos(2));
        glEnd();

        for (int i = 1; i < poses.size(); i++) {
            Twc = poses[i];
            glPointSize(5);
            glBegin(GL_POINTS);
            glColor3f(1, 0, 0);
            glVertex3d(Twc(0,3), Twc(1,3), Twc(2,3));
            glEnd();

            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(pre_pos(0), pre_pos(1), pre_pos(2));
            glVertex3f(Twc(0,3), Twc(1,3), Twc(2,3));
            glEnd();
            pre_pos = Twc.block<3,1>(0,3);
        }

        pangolin::FinishFrame();
        usleep(5000);
    }


}