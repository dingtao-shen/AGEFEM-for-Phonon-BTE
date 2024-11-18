#ifndef MACROPROPERTY
#define MACROPROPERTY

#include <vector>
#include <unordered_map>
#include "Eigen/Dense"
#include "SpatialMesh/data_struct.h"

class MacroProperty
{
private:
    Eigen::VectorXd T;
    Eigen::VectorXd Qx;
    Eigen::VectorXd Qy;
    Eigen::VectorXd T_old;
    Eigen::VectorXd Qx_old;
    Eigen::VectorXd Qy_old;
    Eigen::MatrixXd Ts;
    Eigen::MatrixXd Qxs;
    Eigen::MatrixXd Qys;
    double ResidualT;
public:
    MacroProperty(int Nt, int DoF);
    ~MacroProperty();
    void Cal_MacroProperty(std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Face> Faces, std::vector<Cell> Cells, \
                            Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, Eigen::MatrixXd dOmega, Eigen::MatrixXd IntMat, int Nt, int DoF, \
                            double Cv);
    double Cal_ResidualT();
    Eigen::VectorXd GetT();
    Eigen::VectorXd GetT_old();
    Eigen::MatrixXd GetTs();
    Eigen::MatrixXd GetQxs();
    Eigen::MatrixXd GetQys();
};


unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> Cal_Flux_Wall(std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Face> Faces, std::vector<Cell> Cells, \
                            Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, Eigen::MatrixXd dOmega, Eigen::MatrixXi& bdinf, \
                            std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>>& MassOnFace, \
                            int Np, int Na, int Nf, int DoF3D, int deg);


#endif