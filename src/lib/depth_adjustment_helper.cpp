#include <iostream>

#include "depth_adjustment_helper.hpp"

class OptimizationFunction: public cv::MinProblemSolver::Function
{
public:
    OptimizationFunction(){}
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
        std::cout << "calc func" << std::endl;
        return this->fun(x);
    }

private:
    int dims;
    opt_fun fun;
};


GradientDescent::GradientDescent()
{
    solver = cv::DownhillSolver::create();
    functionPtr = cv::makePtr<OptimizationFunction>();
    solver->setFunction(functionPtr);
}

void GradientDescent::setFunction(opt_fun fun, int dims)
{
    functionPtr.dynamicCast<OptimizationFunction>()->setFunction(fun);
    functionPtr.dynamicCast<OptimizationFunction>()->setDims(dims);
}

void GradientDescent::minimize(std::vector<double> &x0)
{
    solver->minimize(cv::InputOutputArray(x0));
}

void GradientDescent::setInitStep(std::vector<double> &step)
{
    solver->setInitStep(cv::InputArray(step));
}
