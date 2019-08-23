#include <iostream>
#include "opencv2/core/core.hpp"
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

void test()
{
    cout << "test" << endl;
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
        // transform_keypoints(pose, x, 1, fx, fy, cx, cy, kp2d);


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

        kps3d(0, i) = x0[0];
        kps3d(1, i) = x0[1];
        kps3d(2, i) = x0[2];
    }
}
