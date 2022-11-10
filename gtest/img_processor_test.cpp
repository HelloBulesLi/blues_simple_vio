#include <gtest/gtest.h>
#include <opencv/cv.hpp>
#include <fstream>
#include <thread>
#include "msckf_filter.hpp"
#include "img_processor.hpp"
#include <glog/logging.h>
#include <string>
#include <pthread.h>

using namespace std;
using namespace vins_slam;
using namespace testing;

void img_process(std::shared_ptr<ImgProcessor> img_proc_ptr);
void img_read_thread(string &img_path, std::shared_ptr<ImgProcessor> img_proc_ptr);
void imu_read_thread(string &imu_path_cur, std::shared_ptr<ImgProcessor> img_proc_ptr);

typedef struct imu_read_desc {
    string imu_path;
    std::shared_ptr<ImgProcessor> img_proc_ptr;
} imu_read_desc_t;

string data_path = "/media/libo/KINGSTON/Euroc_DataSet/V1_01_easy/mav0/";
string imu_path = data_path+"imu0/";

bool is_first_imu = true;

int main(int argc, char** argv)
{
    InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    return 0;
}

// TEST target:
// 1. display feature detect result
// 2. display feature track result
// 3. calculate the undisort model accurate
// 4. display fundamental matrix remove how much outlier(intliner rate in stereo match)
// 5. display two point ransanc remove how much outlier(inliner rate in track)
// 6. check the two point ransac function, observe the error befor 
// and after outlier removed
// note: the display tool is opencv

TEST(img_process, feature_detect)
{
    std::shared_ptr<ImgProcessor> img_proc(new ImgProcessor());
    std::shared_ptr<OC_MSCKF_Filter> filter_ptr(new OC_MSCKF_Filter());

    img_proc->setMsckf_Filter(filter_ptr);

    std::thread img_proc_handle(img_process, img_proc);
    std::thread img_read_thread_handle(img_read_thread, std::ref(data_path), img_proc);

    std::thread imu_read_thread_handle(imu_read_thread, std::ref(imu_path),img_proc);

    // can used for real time linux  with PREEMPT_RT patch, this could ehance the imu_read_thread's
    // priority 
    /*
    pthread_t imu_read_thread_handle;
    pthread_attr_t imu_read_thread_attr;
    struct sched_param sched_attr;

    int ret = pthread_attr_init(&imu_read_thread_attr);
    assert(ret == 0);

    ret = pthread_attr_setschedpolicy(&imu_read_thread_attr, SCHED_RR);
    assert(ret == 0);

    sched_attr.sched_priority = 95;

    ret = pthread_attr_setschedparam(&imu_read_thread_attr, &sched_attr);
    LOG(INFO) << "current error is " << strerror(errno) << endl;
    assert(ret == 0);

    imu_read_desc_t imu_desc;
    imu_desc.imu_path = imu_path;
    imu_desc.img_proc_ptr = img_proc;

    pthread_create(&imu_read_thread_handle, &imu_read_thread_attr, 
                    imu_read_thread, &imu_desc);
    */

    img_proc_handle.join();
    img_read_thread_handle.join();
    // imu_read_thread_handle.join();
    

}

void img_process(std::shared_ptr<ImgProcessor> img_proc_ptr)
{
    while(true)
    {
        img_proc_ptr->process();
        // waitKey(0);
    }
}

void img_read_thread(string &img_path, std::shared_ptr<ImgProcessor> img_proc_ptr)
{
    string left_img_path = img_path+"cam0/";
    string right_img_path = img_path+"cam1/";
    ifstream left_img_data;
    ifstream right_img_data;
    
    left_img_data.open(left_img_path+"data.csv", ios::in);
    
    if(!left_img_data.is_open())
    {
        LOG(ERROR) << "can't open left img data csv " << endl;
        LOG(ERROR) << "erros is " << strerror(errno) << endl;
        return; 
    }

    right_img_data.open(right_img_path+"data.csv", ios::in);

    if(!right_img_data.is_open())
    {
        LOG(ERROR) << "can't open right img data csv " << endl;
    }

    double first_img_timestamp = 1403715273262140000;
    first_img_timestamp /= 1e9;
    first_img_timestamp  = first_img_timestamp + 1;

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

    LOG(INFO) << "first img measurement timestamp is " << std::setprecision(15) << left_timestamp << endl;

    
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

    int img_cnt = 25;
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

        // for debug
        // LOG(INFO) << "cur frame process time is " <<  cost_time << " ms";
        
        if(cost_time < 50)
        {
            usleep(50*1000 - cost_time*1000);
        }
        auto t3 = std::chrono::steady_clock::now();

        // LOG(INFO) << "cur img data timestamp is " << std::setprecision(15) << left_timestamp << endl;

        // for debug
        LOG(INFO) << "add img interval time is " <<  
                    std::chrono::duration<double, std::milli>(t3-t1).count() << " ms";
    }

    LOG(INFO) << "pub img cnt is " << count << endl;
    while(true)
    {
        usleep(1000*1000);
    }

    return ;
}

void imu_read_thread(string &imu_path_cur, std::shared_ptr<ImgProcessor> img_proc_ptr)
{
    ifstream imu_data;
    imu_data.open(imu_path_cur+"data.csv", ios::in);

    if(!imu_data.is_open())
    {
        LOG(ERROR) << " imu data csv can't open " << endl;
    }

    // double first_imu_timestamp = 1403636580863550000;
    // first_imu_timestamp /= 1e9;
    // first_imu_timestamp = first_imu_timestamp+100;

    double first_imu_timestamp = 1403715273262140000;
    first_imu_timestamp /= 1e9;
    first_imu_timestamp = first_imu_timestamp + 1;

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
   
    int imu_cnt = 4*200;
    int count = 0;

    while(count < imu_cnt)
    {
        if(is_first_imu)
        {
            is_first_imu = false;
            // continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        std::shared_ptr<IMU_Measure> imu(new IMU_Measure(angle_vel, acc_vel, imu_timestamp));
        img_proc_ptr->addimu(imu);

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
        LOG(INFO) << "imu add interval time is " << 
                    std::chrono::duration<double, std::milli>(t3-t1).count() << " ms"<< endl;
    }
    
    // for debug
    LOG(INFO) << "pub imu cnt is " << count << endl;
    while(true)
    {
        usleep(1000*1000);
    }

    return ;
    // return NULL;
    return;
}