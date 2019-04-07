#include "opencv2/core/optim.hpp"
#include <eigen3/Eigen/Dense>

using namespace Eigen;

typedef double (*opt_fun)(void *object, const double* x);

class c_DepthAdjuster
{
public:
    c_DepthAdjuster();

    void adjust_depth(double *new_kps, double* cost);

    void set_image(unsigned char *image, int rows, int cols) {
        this->image = Map<Matrix<unsigned char, Dynamic, Dynamic>>(image, rows, cols);
    }
    void set_new_image(unsigned char *new_image, int rows, int cols) {
        this->new_image = Map<Matrix<unsigned char, Dynamic, Dynamic>>(new_image, rows, cols);
    }
    void set_keypoints3d(double *keypoints3d, int size) {
        this->keypoints3d = Map<Matrix<double, Dynamic, 3>>(keypoints3d, size, 3);
    }
    void set_keypoints2d(double *keypoints2d, int size) {
        this->keypoints2d = Map<Matrix<double, Dynamic, 2>>(keypoints2d, size, 2);
    }
    void set_pose(double* pose) {
        this->pose = Map<Matrix<double, 1, 6>>(pose, 1, 6);
    }

    void set_fx(double fx) {
        this->fx = fx;
    }

    void set_fy(double fy) {
        this->fy = fy;
    }

    void set_cx(double cx) {
        this->cx = cx;
    }

    void set_cy(double cy) {
        this->cy = cy;
    }

    double optimize(const double *x);

private:
    Matrix<unsigned char, Dynamic, Dynamic> image;
    Matrix<unsigned char, Dynamic, Dynamic> new_image;
    Matrix<double, Dynamic, 3> keypoints3d;
    Matrix<double, Dynamic, 2> keypoints2d;
    Matrix<double, 1, 6> pose;

    double fx;
    double fy;
    double cx;
    double cy;

    cv::Ptr<cv::DownhillSolver> solver;
    cv::Ptr<cv::MinProblemSolver::Function> functionPtr;
};

void c_rotation_matrix(double *angle, double* rotation_matrix);
void c_transform_keypoints(double* pose,
        const double* keypoints3d, int number_of_keypoints,
        double fx, double fy, double cx, double cy,
        double *keypoints2d);
double c_get_intensity_diff(const unsigned char *image1,
        const unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        const double *keypoint1,
        const double *keypoint2,
        double errorval);

