#ifndef UTILITIES
#define UTILITIES

#include "Eigen/Dense"
#include "minurbs/minurbs.h"

Eigen::MatrixXd GaussQuad(double a, double b, int N);

Eigen::MatrixXd CrvQuadPts(double r, double start, double end, int n_gl);

Eigen::MatrixXd CellQuadPts(double r, double start, double end, double x_top, double y_top, int deg, int Ngl);

double QuadSum(Eigen::VectorXd f, Eigen::MatrixXd Qpw);

double DivFactorial(int Ns, int Ne, int Ds, int De);

Eigen::MatrixXd StandardBasis(int type, int deg);

Eigen::MatrixXd NodalDis(int deg, double r, double start, double end, double x_interior, double y_interior);

Eigen::MatrixXd cal_Basis(Eigen::MatrixXd NodalDis, int deg);

Eigen::MatrixXd evl_Basis(Eigen::MatrixXd Coef, Eigen::MatrixXd evlPts, int deg);

Eigen::MatrixXd cal_dxBasis(Eigen::MatrixXd Coef, int deg);

Eigen::MatrixXd cal_dyBasis(Eigen::MatrixXd Coef, int deg);

Eigen::MatrixXd evl_dBasis(Eigen::MatrixXd dCoef, Eigen::MatrixXd evlPts, int deg);

#endif