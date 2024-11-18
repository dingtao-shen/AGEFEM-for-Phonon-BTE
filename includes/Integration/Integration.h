#ifndef INTEGRATION
#define INTEGRATION

#include <vector>
#include "Eigen/Dense"
#include "Utility/Utility.h"
#include "VelocityMesh/VelocityMesh.h"
#include "SpatialMesh/SpatialMesh.h"

class Integration{
    private:
        Eigen::MatrixXd IntMat;
        std::vector<Eigen::MatrixXd> MassMat;
        std::vector<Eigen::MatrixXd> StfMatX;
        std::vector<Eigen::MatrixXd> StfMatY;
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> MassOnFace;
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> MassOnFace_in;
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> MassOnFace_out;
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> FluxInt;
        std::vector<std::vector<std::vector<Eigen::MatrixXd>>> IntOnFace; // only compute for straight-sided face
    public:
        Integration(vector<Node>& Nodes, vector<Face>& Faces, vector<Cell>& Cells, Eigen::MatrixXd& dOmega, Eigen::MatrixXd& Cx, Eigen::MatrixXd& Cy, \
                    Eigen::MatrixXi& boundaryinf, int deg, int NPfc, int Nlmd, int Nthe);
        ~Integration();

        Eigen::MatrixXd GetIntMat();
        std::vector<Eigen::MatrixXd> GetMassMat();
        std::vector<Eigen::MatrixXd> GetStfMatX();
        std::vector<Eigen::MatrixXd> GetStfMatY();
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> GetMassOnFace();
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> GetMassOnFace_in();
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> GetMassOnFace_out();
        std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> GetFluxInt();
        std::vector<std::vector<std::vector<Eigen::MatrixXd>>> GetIntOnFace();
};

#endif