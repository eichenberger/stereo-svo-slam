#include <iostream>
#include "slam_accelerator_helper.hpp"

using namespace std;

static Matrix<double, 3, 3> _rot_mat_x(double angle){
    double data[] = {1, 0, 0,
                     0, cos(angle), -sin(angle),
                     0, sin(angle), cos(angle)};
    Matrix<double, 3, 3> m(data);
    return m;
}

static Matrix<double, 3, 3> _rot_mat_y(double angle) {
    double data[] = {cos(angle), 0, sin(angle),
                     0, 1, 0,
                     -sin(angle), 0, cos(angle)};
    Matrix<double, 3, 3> m(data);
    return m;
}

static Matrix<double, 3, 3> _rot_mat_z(double angle) {
    double data[] = {cos(angle), -sin(angle), 0,
                     sin(angle), cos(angle), 0,
                     0, 0, 1};
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
    // We need RowMajor here, because c_rotation_matrix is also used by numpy
    Matrix<double, 3, 3, RowMajor> rotation_matrix;
    // Calculate the extrinsic from pose
    c_rotation_matrix(&pose[3], rotation_matrix.data());
    extrinsic.topLeftCorner<3,3>() =  rotation_matrix;
    Map<Vector3d> pose_vector(pose);
    extrinsic.rightCols(1) = pose_vector;

    // Add a 4th row with ones at the end of the vector
    Matrix<double, 4, Dynamic> kps3d;
    kps3d.setOnes(4, number_of_keypoints);

    // Numpy expects RowMajor layout of the data
    Map<Matrix<double, 3, Dynamic, RowMajor>> temp_kps((double*)keypoints3d, 3, number_of_keypoints);
    kps3d.topLeftCorner(3, number_of_keypoints) = temp_kps;

    Matrix3d intrinsic;
    intrinsic << fx,0,cx, 0, fy, cy, 0, 0, 1;

    //cout << "rotationmatrix: " << rotation_matrix << endl;
    //cout << "Intrinsic: " << intrinsic << endl;
    //cout << "Extrinsic: " << extrinsic << endl;

    // Numpy expects RowMajor layout of the data
    Map<Matrix<double, 3, Dynamic, RowMajor>> kps2d(keypoints2d, 3, number_of_keypoints);
    kps2d = intrinsic * extrinsic * kps3d;

    // Fix scaling
    kps2d = kps2d.array() / kps2d.bottomRows(1).array().replicate(3,1);
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

    double x_ceil_prob =  1.0 - x_floor_prob;
    double y_ceil_prob =  1.0 - y_floor_prob;

    double sub_pixel_val = 0.0;

    sub_pixel_val = x_floor_prob*y_floor_prob*image[y_floor*image_width + x_floor];
    sub_pixel_val += x_floor_prob*y_ceil_prob*image[y_ceil*image_width + x_floor];
    sub_pixel_val += x_ceil_prob*y_floor_prob*image[y_floor*image_width + x_ceil];
    sub_pixel_val += x_ceil_prob*y_ceil_prob*image[y_ceil*image_width + x_ceil];

    return sub_pixel_val;
}

void test()
{
    cout << "test" << endl;
}
double c_get_intensity_diff(const unsigned char *image1,
        const unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        const double *keypoint1,
        const double *keypoint2,
        double errorval)
{
    const int MEASUREMENT_DISTANCE = 12;

    double x1 = keypoint1[0];
    double y1 = keypoint1[1];
    double x2 = keypoint2[0];
    double y2 = keypoint2[1];

    // If keypoint is outside of image ignore it
    if ((x1 - MEASUREMENT_DISTANCE < 0) ||
            (x1 + MEASUREMENT_DISTANCE > image_width) ||
            (y1 - MEASUREMENT_DISTANCE < 0) ||
            (y1 + MEASUREMENT_DISTANCE > image_height))
        return errorval;

    if ((x2 - MEASUREMENT_DISTANCE < 0) ||
            (x2 + MEASUREMENT_DISTANCE > image_width) ||
            (y2 - MEASUREMENT_DISTANCE < 0) ||
            (y2 + MEASUREMENT_DISTANCE > image_height))
        return errorval;


    Matrix<double, 5,5> block1;
    Matrix<double, 5,5> block2;

    {
        double y,x;

        // block in image 1
        y = y1-2; x= x1-2; block1(0,0) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-1; x= x1-2; block1(1,0) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-0; x= x1-2; block1(2,0) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+1; x= x1-2; block1(3,0) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+2; x= x1-2; block1(4,0) = c_get_sub_pixel(image1, image_width, image_height, x, y);

        y = y1-2; x= x1-1; block1(0,1) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-1; x= x1-1; block1(1,1) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-0; x= x1-1; block1(2,1) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+1; x= x1-1; block1(3,1) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+2; x= x1-1; block1(4,1) = c_get_sub_pixel(image1, image_width, image_height, x, y);

        y = y1-2; x= x1-0; block1(0,2) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-1; x= x1-0; block1(1,2) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-0; x= x1-0; block1(2,2) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+1; x= x1-0; block1(3,2) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+2; x= x1-0; block1(4,2) = c_get_sub_pixel(image1, image_width, image_height, x, y);


        y = y1-2; x= x1+1; block1(0,3) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-1; x= x1+1; block1(1,3) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-0; x= x1+1; block1(2,3) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+1; x= x1+1; block1(3,3) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+2; x= x1+1; block1(4,3) = c_get_sub_pixel(image1, image_width, image_height, x, y);


        y = y1-2; x= x1+2; block1(0,4) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-1; x= x1+2; block1(1,4) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1-0; x= x1+2; block1(2,4) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+1; x= x1+2; block1(3,4) = c_get_sub_pixel(image1, image_width, image_height, x, y);
        y = y1+2; x= x1+2; block1(4,4) = c_get_sub_pixel(image1, image_width, image_height, x, y);
    }

    // block in image 2
    {
        double y,x;

        y = y2-2; x= x2-2; block2(0,0) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-1; x= x2-2; block2(1,0) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-0; x= x2-2; block2(2,0) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+1; x= x2-2; block2(3,0) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+2; x= x2-2; block2(4,0) = c_get_sub_pixel(image2, image_width, image_height, x, y);

        y = y2-2; x= x2-1; block2(0,1) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-1; x= x2-1; block2(1,1) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-0; x= x2-1; block2(2,1) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+1; x= x2-1; block2(3,1) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+2; x= x2-1; block2(4,1) = c_get_sub_pixel(image2, image_width, image_height, x, y);

        y = y2-2; x= x2-0; block2(0,2) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-1; x= x2-0; block2(1,2) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-0; x= x2-0; block2(2,2) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+1; x= x2-0; block2(3,2) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+2; x= x2-0; block2(4,2) = c_get_sub_pixel(image2, image_width, image_height, x, y);


        y = y2-2; x= x2+1; block2(0,3) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-1; x= x2+1; block2(1,3) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-0; x= x2+1; block2(2,3) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+1; x= x2+1; block2(3,3) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+2; x= x2+1; block2(4,3) = c_get_sub_pixel(image2, image_width, image_height, x, y);


        y = y2-2; x= x2+2; block2(0,4) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-1; x= x2+2; block2(1,4) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2-0; x= x2+2; block2(2,4) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+1; x= x2+2; block2(3,4) = c_get_sub_pixel(image2, image_width, image_height, x, y);
        y = y2+2; x= x2+2; block2(4,4) = c_get_sub_pixel(image2, image_width, image_height, x, y);
    }

    double diff = (block2-block1).lpNorm<1>();

    return diff;
}


void c_get_total_intensity_diff(const unsigned char *image1,
        const unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        const double *keypoints1,
        const double *keypoints2,
        unsigned int n_keypoints,
        double *diff)
{
#pragma omp parallel for
    for (unsigned int i = 0; i < n_keypoints; i++) {
        double kp1[] = {keypoints1[i], keypoints1[n_keypoints+i]};
        double kp2[] = {keypoints2[i], keypoints2[n_keypoints+i]};
        diff[i] = c_get_intensity_diff(image1, image2,
                image_width, image_height,
                kp1, kp2, 0);
    }
}

class CloudRefiner : public cv::MinProblemSolver::Function
{
public:
    CloudRefiner(double fx, double fy, double cx, double cy,
            double *pose, Vector2d keypoint2d):
        fx(fx), fy(fy), cx(cx), cy(cy),
        pose(pose), keypoint2d(keypoint2d)
    {}

    double calc(const double* x) const
    {
        double kp2d[3];
        c_transform_keypoints(pose, x, 1, fx, fy, cx, cy, kp2d);


        Vector2d kp2d_map(kp2d);

        Vector2d diff = kp2d_map - keypoint2d;
        return diff.dot(diff);
    }

    int getDims() const
    {
        return 3;
    }

private:
    double fx;
    double fy;
    double cx;
    double cy;

    double *pose;
    Vector2d keypoint2d;
};


void c_refine_cloud(double fx,
        double fy,
        double cx,
        double cy,
        double *pose,
        double *keypoints3d,
        double *keypoints2d,
        int number_of_keypoints)
{
    cv::Ptr<cv::DownhillSolver> solver;
    cv::Ptr<CloudRefiner> refiner;
    Map<Matrix<double, 2, Dynamic, RowMajor>> kps2d(keypoints2d, 2, number_of_keypoints);
    Map<Matrix<double, 3, Dynamic, RowMajor>> kps3d(keypoints3d, 3, number_of_keypoints);
    // Order is x,y,z of 3d point and x,y in original 2d image.
    // We only want to modify z
    double init_steps[] = {0.1,0.1,0.1};
    double cost[number_of_keypoints];
    for (int i = 0; i < number_of_keypoints; i++) {
        // For each point we need to solve the keypoint problem
        double x0[] = {
            kps3d(0, i),
            kps3d(1, i),
            kps3d(2, i)
        };
        solver = cv::DownhillSolver::create();
        refiner = new CloudRefiner(fx, fy, cx, cy, pose, kps2d.col(i));
        solver->setFunction(refiner);
        solver->setInitStep(cv::Mat(3, 1, CV_64F, init_steps));

        cost[i] = solver->minimize(cv::Mat(3,1, CV_64F, x0));
        kps3d(0, i) = x0[0];
        kps3d(1, i) = x0[1];
        kps3d(2, i) = x0[2];
    }
}
