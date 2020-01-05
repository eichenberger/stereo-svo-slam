#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

/*!
 * \brief Class that estimates depth (internal use)
 */
class DepthCalculator
{
public:
    DepthCalculator(){}
    void calculate_depth(Frame &frame,
            const struct CameraSettings &camera_settings);

};

