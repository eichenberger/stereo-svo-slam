#include "opencv2/core/optim.hpp"

typedef double (*opt_fun)(const double* x);

class GradientDescent
{
public:
    GradientDescent();
    void setFunction(opt_fun fun, int dims);
    void minimize(std::vector<double> &x0);
    void setInitStep(std::vector<double> &step);

private:
    cv::Ptr<cv::DownhillSolver> solver;
    cv::Ptr<cv::MinProblemSolver::Function> functionPtr;
};


