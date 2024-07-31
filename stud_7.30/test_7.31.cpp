#define _CRT_SECURE_NO_WARNINGS 1

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <gnuplot-iostream.h>

// 使用Eigen库进行多项式拟合
Eigen::VectorXd polyfit(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int degree) {
    Eigen::MatrixXd X(x.size(), degree + 1);
    for (int i = 0; i < degree + 1; ++i) {
        X.col(i) = x.pow(i);
    }
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd yT = X.transpose() * y;
    Eigen::VectorXd coefficients = XtX.colPivHouseholderQr().solve(yT);
    return coefficients;
}

int main() {
    // 定义x和y数据
    std::vector<double> xi = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<double> yi = { -4.3, -7, -4.7, 4, 23.7, 56.6, 105.2, 176.1, 304.0, 429.7 };

    // 转换为Eigen矩阵
    Eigen::Map<Eigen::VectorXd> x(xi.data(), xi.size());
    Eigen::Map<Eigen::VectorXd> y(yi.data(), yi.size());

    // 多项式拟合
    int degree = 4;
    Eigen::VectorXd coefficients = polyfit(x, y, degree);

    // 创建用于绘图的gnuplot对象
    Gnuplot gp;
    gp << "set terminal png\n";
    gp << "set output 'fit.png'\n";

    // 绘制原始数据点
    gp << "plot '-' with points pointtype 7\n";
    for (size_t i = 0; i < xi.size(); ++i) {
        gp << xi[i] << " " << yi[i] << "\n";
    }
    gp << "e\n";

    // 绘制拟合曲线
    gp << "replot '-' with lines\n";
    for (double t = 1; t <= 10; ++t) {
        double y_fit = coefficients(0);
        for (int i = 1; i <= degree; ++i) {
            y_fit += coefficients(i) * pow(t, i);
        }
        gp << t << " " << y_fit << "\n";
    }
    gp << "e\n";
    return 0;
}

double linear_interpolation(double x, std::vector<double> xi, std::vector<double> yi) {
    int n = xi.size();
    int i = 0;
    while (i < n && x >= xi[i]) ++i;
    if (i == 0 || i == n) throw std::out_of_range("x out of range");
    double slope = (yi[i] - yi[i - 1]) / (xi[i] - xi[i - 1]);
    return yi[i - 1] + slope * (x - xi[i - 1]);
}

// 使用方法
double x_new = 2.5;
double y_new = linear_interpolation(x_new, xi, yi);


#include <Eigen/Dense>
#include <gnuplot-iostream.h>
#include <vector>

Eigen::VectorXd polyfit(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int degree) {
    Eigen::MatrixXd X(x.size(), degree + 1);
    for (int i = 0; i <= degree; ++i) {
        X.col(i) = x.array().pow(i);
    }
    Eigen::VectorXd coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    return coefficients;
}

double polyval(const Eigen::VectorXd& coef, double x) {
    double y = 0;
    for (int i = 0; i < coef.size(); ++i) {
        y += coef[i] * pow(x, i);
    }
    return y;
}

int main() {
    std::vector<double> xi = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<double> yi = { -4.3, -7, -4.7, 4, 23.7, 56.6, 105.2, 176.1, 304.0, 429.7 };

    Eigen::VectorXd x(xi.begin(), xi.end()), y(yi.begin(), yi.end());
    int degree = 4;
    Eigen::VectorXd coefficients = polyfit(x, y, degree);

    Gnuplot gp;
    gp << "set terminal png\n";
    gp << "set output 'polynomial_fit.png'\n";
    gp << "plot '-' with points pointtype 7, '-' with lines\n";

    // Plot original data points
    for (size_t i = 0; i < xi.size(); ++i) {
        gp << xi[i] << " " << yi[i] << "\n";
    }
    gp << "e\n";

    // Plot polynomial fit
    for (double xval = 1.0; xval < 10.0; xval += 0.1) {
        double yval = polyval(coefficients, xval);
        gp << xval << " " << yval << "\n";
    }
    gp << "e\n";
    return 0;
}


#include <Gnuplot-iostream.h>
#include <GSL/gsl_multifit.h>
#include <cmath>
#include <iostream>
#include <vector>

double f(const std::vector<double>& params, double t) {
    double A = params;
    double B = params[1];
    double C = params[2];
    return A * exp(B * t + C * t * t);
}

int main() {
    std::vector<double> x = { 0, 27, 40, 52, 70, 89, 106 };
    std::vector<double> y = { 100, 82.2, 76.3, 71.8, 66.4, 63.3, 61.3 };
    std::vector<double> coefs0 = { 1, 0, 0 };

    // Convert to GSL format
    gsl_vector* x_gsl = gsl_vector_alloc(x.size());
    gsl_vector* y_gsl = gsl_vector_alloc(y.size());
    gsl_matrix* cov = gsl_matrix_alloc(3, 3);
    gsl_multifit_function f_fit;
    f_fit.n = x.size();
    f_fit.p = 3;
    f_fit.params = coefs0.data();
    f_fit.f = &f;
    f_fit.df = nullptr;
    f_fit.fdf = nullptr;

    // Fill GSL vectors
    for (size_t i = 0; i < x.size(); ++i) {
        gsl_vector_set(x_gsl, i, x[i]);
        gsl_vector_set(y_gsl, i, y[i]);
    }

    // Perform nonlinear fit
    gsl_vector* c = gsl_vector_alloc(3);
    gsl_multifit_fsolver_type* T = gsl_multifit_fsolver_lmsder;
    gsl_multifit_fsolver* s = gsl_multifit_fsolver_alloc(T, f_fit.n, f_fit.p);
    gsl_multifit_fsolver_set(s, &f_fit, coefs0.data(), nullptr);
    int status;
    size_t iter = 0;
    do {
        iter++;
        status = gsl_multifit_fsolver_iterate(s);
        if (status) {
            break;
        }
        status = gsl_multifit_fsolver_test_delta(s->dx, s->x, 1e-4, 1e-4);
    } while (status == GSL_CONTINUE && iter < 1000);

    gsl_vector* ycal_gsl = gsl_vector_alloc(x_gsl->size);
    for (size_t i = 0; i < x_gsl->size; ++i) {
        double xval = gsl_vector_get(x_gsl, i);
        double ycal = f(s->x, xval);
        gsl_vector_set(ycal_gsl, i, ycal);
    }

    // Convert GSL results back to std::vector for plotting
    std::vector<double> sol(s->x->data, s->x->data + 3);
    std::vector<double> ycal(ycal_gsl->data, ycal_gsl->data + ycal_gsl->size);

    // Plot data and fitted curve
    Gnuplot gp;
    gp << "set terminal png\n";
    gp << "set output 'nonlinear_fit.png'\n";
    gp << "plot '-' with points pointtype 7 title 'Data', '-' with lines title 'Fit'\n";

    // Plot original data points
    for (size_t i = 0; i < x.size(); ++i) {
        gp << x[i] << " " << y[i] << "\n";
    }
    gp << "e\n";

    // Plot fit curve
    for (size_t i = 0; i < x.size(); ++i) {
        gp << x[i] << " " << ycal[i] << "\n";
    }
    gp << "e\n";

    // Clean up
    gsl_vector_free(x_gsl);
    gsl_vector_free(y_gsl);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    gsl_multifit_fsolver_free(s);
    gsl_multifit_fsolver_type_free(T);
    gsl_vector_free(ycal_gsl);

    return 0;
}