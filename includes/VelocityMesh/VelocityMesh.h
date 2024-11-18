#ifndef VELOCITYMESH
#define VELOCITYMESH

#include <vector>
#include "Eigen/Dense"

class VelocityMesh
{
private:
    Eigen::MatrixXd dOmega;
    Eigen::MatrixXd Cx;
    Eigen::MatrixXd Cy;
public:
    VelocityMesh(int NP, int NA, double VG, double CV);
    ~VelocityMesh();
    Eigen::MatrixXd GetdOmega();
    Eigen::MatrixXd GetCx();
    Eigen::MatrixXd GetCy();
};

#endif