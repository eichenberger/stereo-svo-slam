#include <iostream>
#include "depth_adjustment_helper.hpp"

using namespace std;

class OptimizationFunction: public cv::MinProblemSolver::Function
{
public:
    OptimizationFunction(void *object):object(object)
    {}
    void setFunction(opt_fun fun)
    {
        this->fun = fun;
    }
    void setDims(int dims)
    {
        this->dims = dims;
    }
    int getDims() const
    {
        return dims;
    }

    double calc(const double* x) const
    {
        return this->fun(this->object, x);
    }

private:
    int dims;
    void *object;
    opt_fun fun;
};

static double optimize_depth(void *object, const double *x)
{
    c_DepthAdjuster *adjuster = (c_DepthAdjuster*) object;
    return adjuster->optimize(x);
}

c_DepthAdjuster::c_DepthAdjuster()
{
    solver = cv::DownhillSolver::create();
    functionPtr = cv::makePtr<OptimizationFunction>(this);
    functionPtr.dynamicCast<OptimizationFunction>()->setFunction(optimize_depth);
    functionPtr.dynamicCast<OptimizationFunction>()->setDims(5);
    solver->setFunction(functionPtr);
    // Order is x,y,z of 3d point and x,y in original 2d image.
    // We only want to modify z
    double init_setps[] = {0,0,0.1,0,0,0};
    solver->setInitStep(cv::Mat(5, 1, CV_64F, init_setps));
}

double c_DepthAdjuster::optimize(const double *x)
{
    double kp2d[3];
    c_transform_keypoints(pose.data(), x, 1, fx, fy, cx, cy, kp2d);

    double diff = c_get_intensity_diff(image.data(),
                              new_image.data(),
                              image.cols(),
                              image.rows(),
                              &x[3],
                              kp2d,
                              100000000000.0);

    return diff;

}

void c_DepthAdjuster::adjust_depth(double *new_z, double *cost)
{
    for (int i = 0; i < keypoints3d.rows(); i++) {
        double x0[] = {
            keypoints3d(i,0),
            keypoints3d(i,1),
            keypoints3d(i,2),
            this->keypoints2d(i,0),
            this->keypoints2d(i,1),
        };
        cost[i] = solver->minimize(cv::Mat(5,1, CV_64F, x0));
        new_z[i] = x0[2];
    }
}

static Matrix<double, 3, 3> _rot_mat_x(double angle){
    double data[] = {1, 0, 0, 0, cos(angle), -sin(angle), 0, sin(angle), cos(angle)};
    Matrix<double, 3, 3> m(data);
    return m;
}

static Matrix<double, 3, 3> _rot_mat_y(double angle) {
    double data[] = {cos(angle), 0, sin(angle), 0, 1, 0, -sin(angle), 0, cos(angle)};
    Matrix<double, 3, 3> m(data);
    return m;
}

static Matrix<double, 3, 3> _rot_mat_z(double angle) {
    double data[] = {cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1};
    Matrix<double, 3, 3> m(data);
    return m;
}

void c_rotation_matrix(double *angle, double* rotation_matrix)
{
    Matrix<double, 3, 3> rot_x = _rot_mat_x(angle[0]);
    Matrix<double, 3, 3> rot_y = _rot_mat_y(angle[1]);
    Matrix<double, 3, 3> rot_z = _rot_mat_z(angle[2]);


    Map<Matrix3d> rot (rotation_matrix);
    rot = rot_x*rot_y*rot_z;
}

void c_transform_keypoints(double* pose,
        const double* keypoints3d, int number_of_keypoints,
        double fx, double fy, double cx, double cy,
        double *keypoints2d)
{
    Matrix<double, 3, 4> extrinsic;
    Matrix<double, 3, 3> rotation_matrix;
    // Calculate the extrinsic from pose
    c_rotation_matrix(&pose[3], rotation_matrix.data());
    extrinsic.topLeftCorner<3,3>() =  rotation_matrix;
    Map<Vector3d> pose_vector(pose);
    extrinsic.rightCols(1) = pose_vector;

    Matrix<double, 4, Dynamic> kps3d;
    kps3d.setOnes(4, number_of_keypoints);

    Map<MatrixXd> temp_kps((double*)keypoints3d, 3, number_of_keypoints);
    kps3d.topLeftCorner(3, number_of_keypoints) = temp_kps;

    // Numpy expects RowMajor layout of the data
    Map<Matrix<double, 3, Dynamic, RowMajor>> kps2d(keypoints2d, 3, number_of_keypoints);
    kps2d = extrinsic * kps3d;

    double _focal[] = {fx, fy, 1};
    Map<Vector3d> focal(_focal);
    double _pp[] = {cx, cy, 0};
    Map<Vector3d> principal_point(_pp);

    kps2d = kps2d.array() * focal.array().replicate(1, number_of_keypoints);
    kps2d = kps2d.array() / kps2d.bottomRows(1).array().replicate(3,1);
    kps2d = kps2d.array() + principal_point.array().replicate(1, number_of_keypoints);
}

static double c_get_sub_pixel(const unsigned char *image,
        unsigned int image_width,
        unsigned int image_height,
        double x,
        double y)
{
    int x_floor = int(x);
    int y_floor = int(y);

    int x_ceil = int(x+1);
    int y_ceil = int(y+1);

    double x_floor_prob =  1.0 - (x - x_floor);
    double y_floor_prob =  1.0 - (y - y_floor);

    double x_ceil_prob =  1.0 - (x_ceil - x);
    double y_ceil_prob =  1.0 - (y_ceil - y);

    double sub_pixel_val = 0.0;

    sub_pixel_val = x_floor_prob*y_floor_prob*image[y_floor*image_width + x_floor];
    sub_pixel_val += x_floor_prob*y_ceil_prob*image[y_ceil*image_width + x_floor];
    sub_pixel_val += x_ceil_prob*y_floor_prob*image[y_floor*image_width + x_ceil];
    sub_pixel_val += x_ceil_prob*y_ceil_prob*image[y_ceil*image_width + x_ceil];

    return sub_pixel_val;
}


double c_get_intensity_diff(const unsigned char *image1,
        const unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        const double *keypoint1,
        const double *keypoint2,
        double errorval)
{
    const int MEASUREMENT_DISTANCE = 5;

    int x1 = int(keypoint1[0]);
    int y1 = int(keypoint1[1]);
    double x2 = keypoint2[0];
    double y2 = keypoint2[1];

    // If keypoint is outside of second image we ignore it
    if ((x2 - MEASUREMENT_DISTANCE < 0) ||
            (x2 + MEASUREMENT_DISTANCE > image_width) ||
            (y2 - MEASUREMENT_DISTANCE < 0) ||
            (y2 + MEASUREMENT_DISTANCE > image_height))
        return errorval;

    double diff = 0;

//    diff =  fabs(image1[y1*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2));
//    diff += fabs(image1[y1*image_width + x1-1] - c_get_sub_pixel(image2, image_width, image_height, x2-1, y2));
//    diff += fabs(image1[(y1-1)*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2-1));
//    diff += fabs(image1[y1*image_width + x1+1] - c_get_sub_pixel(image2, image_width, image_height, x2+1, y2));
//    diff += fabs(image1[(y1+1)*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2+1));



    diff =  fabs(image1[y1*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2));
    diff += fabs(image1[y1*image_width + x1-MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2-MEASUREMENT_DISTANCE, y2));
    diff += fabs(image1[(y1-MEASUREMENT_DISTANCE)*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2-MEASUREMENT_DISTANCE));
    diff += fabs(image1[y1*image_width + x1+MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2+MEASUREMENT_DISTANCE, y2));
    diff += fabs(image1[(y1+MEASUREMENT_DISTANCE)*image_width + x1] - c_get_sub_pixel(image2, image_width, image_height, x2, y2+MEASUREMENT_DISTANCE));

    diff +=  fabs(image1[(y1-MEASUREMENT_DISTANCE)*image_width + x1-MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2-MEASUREMENT_DISTANCE, y2-MEASUREMENT_DISTANCE));
    diff +=  fabs(image1[(y1-MEASUREMENT_DISTANCE)*image_width + x1+MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2+MEASUREMENT_DISTANCE, y2-MEASUREMENT_DISTANCE));

    diff +=  fabs(image1[(y1+MEASUREMENT_DISTANCE)*image_width + x1-MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2-MEASUREMENT_DISTANCE, y2+MEASUREMENT_DISTANCE));
    diff +=  fabs(image1[(y1+MEASUREMENT_DISTANCE)*image_width + x1+MEASUREMENT_DISTANCE] - c_get_sub_pixel(image2, image_width, image_height, x2+MEASUREMENT_DISTANCE, y2+MEASUREMENT_DISTANCE));

    return diff;
}
