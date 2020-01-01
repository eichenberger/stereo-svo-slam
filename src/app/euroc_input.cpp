#include <regex>

#include "euroc_input.hpp"

using namespace std;
using namespace cv;


EurocInput::EurocInput(const std::string &image_path,
            const std::string &settings) :
    read_count(0)
{
    read_settings(settings);
    load_images(image_path);


    // Read rectification parameters
    cv::FileStorage fsSettings(settings, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
    }

    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);


}

void EurocInput::get_camera_settings(CameraSettings &camera_settings)
{
    camera_settings = this->camera_settings;
}

bool EurocInput::read(cv::Mat &left, cv::Mat &right, float &time_stamp)
{
    Mat image;

    if (read_count >= left_images.size())
        return false;
    Mat _left = cv::imread(left_images[read_count]);
    Mat _right = cv::imread(right_images[read_count]);

    Mat __left, __right;
    cv::remap(_right,__right,M1l,M2l,cv::INTER_LINEAR);
    cv::remap(_left,__left,M1r,M2r,cv::INTER_LINEAR);

    cvtColor(__left, left, COLOR_BGR2GRAY);
    cvtColor(__right, right, COLOR_BGR2GRAY);

    time_stamp = timestamps[read_count];

    read_count++;
    return true;
}

void EurocInput::jump_to(int frame_number)
{
    read_count = frame_number;
}


void EurocInput::load_images(string image_path)
{
    ifstream data_file;
    data_file.open(image_path + "cam0/data.csv");
    double t0 = -1.0;
    while(!data_file.eof())
    {
        string s;
        getline(data_file,s);
        if(!s.empty() && s[0] != '#')
        {
            string file_name = regex_replace(s, regex(".*,"), "");
            file_name = regex_replace(file_name, regex("\r"), "");
            right_images.push_back(image_path + "cam0/data/" + file_name);
            left_images.push_back(image_path + "cam1/data/" + file_name);

            string timestamp_string = regex_replace(s, regex(",.*"), "");
            timestamp_string = regex_replace(timestamp_string, regex("\r"), "");
            double t = stod(timestamp_string)/1.0e9;
            if (t0 < 0.0)
                t0 = t;

            t = t-t0;
            timestamps.push_back((float)t);
        }
    }
}
