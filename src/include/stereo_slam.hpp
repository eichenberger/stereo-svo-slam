#ifndef STEREO_SLAM_HPP
#define STEREO_SLAM_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"
#include "keyframe_manager.hpp"

/*!
 * \brief Class for the whole SVO Stereo SLAM
 *
 * This is the class used to create a SVO Stereo SLAM object. It is the main
 * class used to interact with in user programs. Creating the object requires
 * valid camera settings. After that it can be fed with new images
 */
class StereoSlam
{
public:
    /*!
     * \brief Create the stereo SLAM object
     *
     * @param[in] camera_settings The camera settings that the define the camera in use
     */
    StereoSlam(const CameraSettings &camera_settings);

    /*!
     * \brief Process new image from the camera
     *
     * @param[in] left Left stereo image (from behind the camera)
     * @param[in] right Right stereo image (from behind the camera)
     * @param[in] time_stamp Time of when the image was taken in seconds (float)
     */
    void new_image(const cv::Mat &left, const cv::Mat &right, const float time_stamp);

    void get_keyframe(KeyFrame &keyframe); //!< Receive last keyframe
    void get_keyframes(std::vector<KeyFrame> &keyframes);   //!< Receive all keyframes
    bool get_frame(Frame &frame);   //!< Recevie frame
    void get_trajectory(std::vector<Pose> &trajectory); //!< Receive trajectory for all frames

    /*!
     * \brief Update the camera pose externally
     *
     * @param[in] pose A vector describeing the measured pose
     * @param[in] speed A vector describeing the measured speed (velocity) in x,y,z and rx,ry,rz direction
     * @param[in] pose_variance The variance of the pose measurement
     * @param[in] speed_variance The variance of the speed measurement
     * @param[in] dt Time since last update (1/f)
     */
    Pose update_pose(const Pose &pose, const cv::Vec6f &speed,
        const cv::Vec6f &pose_variance, const cv::Vec6f &speed_variance,
        double dt);

private:
    void new_keyframe();

    void estimate_pose(Frame *previous_frame);
    void remove_outliers(Frame *frame);

    const CameraSettings camera_settings;
    KeyFrameManager keyframe_manager;
    KeyFrame* keyframe;
    cv::Ptr<Frame> frame;
    std::vector<Pose> trajectory;
    cv::Vec6f motion;

    cv::KalmanFilter kf;
    cv::TickMeter time_measure;
};

#endif
