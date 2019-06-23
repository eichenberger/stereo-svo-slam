#include "rotation_matrix.hpp"

#include <cmath>

using namespace cv;
using namespace std;

static Matx33f _rot_mat_x(float angle){
    Matx33f m({1, 0, 0,
            0, cosf(angle), -sinf(angle),
            0, sinf(angle), cosf(angle)});
    return m;
}

static Matx33f _rot_mat_y(float angle) {
    Matx33f m({cosf(angle), 0, sinf(angle),
            0, 1, 0,
            -sinf(angle), 0, cosf(angle)});
    return m;
}

static Matx33f _rot_mat_z(float angle) {
    Matx33f m({cosf(angle), -sinf(angle), 0,
            sinf(angle), cosf(angle), 0,
            0, 0, 1});
    return m;
}

void rotation_matrix(const vector<float> &angle, Matx33f &rotation_matrix)
{
    Matx33f rot_x = _rot_mat_x(angle[0]);
    Matx33f rot_y = _rot_mat_y(angle[1]);
    Matx33f rot_z = _rot_mat_z(angle[2]);


    rotation_matrix = rot_x.mul(rot_y).mul(rot_z);
}
