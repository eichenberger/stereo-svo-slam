#ifndef STEREO_SLAM_TYPES_H
#define STEREO_SLAM_TYPES_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "pose_manager.hpp"

/*!
 * \brief Structure representing the settings for the SLAM algorithm for a
 * specific camera type.
 *
 * Each camera works best with it's own settings. Some parameters like
 * fx, fy, etc. are physicaly given, some are not (e.g. grid_height)
 */
struct CameraSettings {
    float baseline; //!< Baseline of the camera in meter times fx
    float fx;   //!< Focal length in pixels along x axis
    float fy;   //!< Focal length in pixels along y axis
    float cx;   //!< Camera principal point along x axis
    float cy;   //!< Camera principal point along y axis
    float k1;   //!< Distortion parameter k1 (radial 1)
    float k2;   //!< Distortion parameter k2 (radial 2)
    float k3;   //!< Distortion parameter k3 (radial 3)
    float p1;   //!< Distortion parameter p1 (tangential 1)
    float p2;   //!< Distortion parameter p2 (tangential 2)
    int grid_height;    //!< Grid height in pixels (for keypoint detection)
    int grid_width;     //!< Grid width in pixels (for keypoint detection)
    int search_x;       //!< Depth calculator maximum disparity
    int search_y;       //!< Depth calculator maximum missalignment in y direction
    int window_size_pose_estimator;   //!< Window size for pose estimation (4 works okay)
    int window_size_opt_flow;   //!< Window size for optical flow (31 works okay)
    int window_size_depth_calculator;   //!< Window size for depth calculator (31 works okay)
    int max_pyramid_levels; //!< Maximum pyramid levels for pose estimation
    int min_pyramid_level_pose_estimation;  //!< Minimum pyramid level for pose estimation (e.g. if max =3 and min=2 it wont search on level 1)
};

/*!
 * \brief Structure used to store a stereo image (internal use )
 */
struct StereoImage {
    std::vector<cv::Mat> left;  //!< Left image pyramid (max_pyramid_levels), left[0] is the orignal
    std::vector<cv::Mat> right; //!< Right image pyramid (currently only one level!), right[0] is the original
    std::vector<cv::Mat> opt_flow;  //!< Optical flow pyramid of left image used only for optical flwo (different from other pyramids)
};


/*!
 * \brief Keypoint type edgled or fast corner
 */
enum KeyPointType {
    KP_FAST,    //!< Keypoint is a fast corner
    KP_EDGELET  //!< Keypoint is an edglet
};

/*!
 * \brief A 2D keypoint representation in the image
 */
struct KeyPoint2d {
    float x;    //!< X position
    float y;    //!< Y position
};

/*!
 * \brief A 3D keypoint representation in global coordinates
 */
struct KeyPoint3d {
    float x;    //!< X position
    float y;    //!< Y position
    float z;    //!< Z position
};

/*!
 * \brief Color representation
 */
struct Color {
    uint8_t r;  //!< red
    uint8_t g;  //!< green
    uint8_t b;  //!< blue
};

/*!
 * \brief Structure holding information about a keypoint
 */
struct KeyPointInformation {
    float score;    //!< Score achieved during keypoint detection
    int level;  //!< Pyramid level on which keypoint was detected
    enum KeyPointType type; //!< The keypoint type
    uint64_t keyframe_id;   //!< The keyframe id where the keypoint was found
    size_t keypoint_index;  //!< The keypoint index in the keyframe
    Color color;    //!< The color of the keypoint (for debugging)
    bool ignore_during_refinement;  //!< Should be ignored during refinement
    bool ignore_completely; //!< Should be ignored completely (will be removed in next run)
    int outlier_count;  //!< The count the point is rates as outlier so far
    int inlier_count;   //!< The count the point is rated as inlier so far
    bool ignore_temporary;  //!< Ignore the keypoint temporary
    cv::KalmanFilter kf;    //!< Kalman filter used for depth filter including parameters
};

/*!
 * \brief Collection of keypoints
 *
 * Each entry has the same index. We try to avaoid mixing information to speed
 * up calculation (e.g. kp2d<->kp3d mixing). The size of all elements should be the same.
 */
struct KeyPoints {
    std::vector<KeyPoint2d> kps2d;  //!< 2D keypoints
    std::vector<KeyPoint3d> kps3d;  //!< 3D keypoints
    std::vector<KeyPointInformation> info;  //!< Information about the keypoint
};

/*!
 * \brief Representation of a frame
 *
 * This includes pose, keypoints, stereo image, etc.
 */
struct Frame {
    uint64_t id;    //!< The frame id
    PoseManager pose;   //!< Pose assigned to the frame
    struct StereoImage stereo_image;    //!< Stereo image assigned to the framek
    struct KeyPoints kps;   //!< Keypoints used by the frame
    double time_stamp;  //!< Timestamp
};

/*!
 * \brief A keyframe is a frame which inserts new keypoints
 *
 * It's basically a normal frame with a new id. The algorithm decides when to insert  a new frame.
 */
struct KeyFrame : Frame{
};


#endif
