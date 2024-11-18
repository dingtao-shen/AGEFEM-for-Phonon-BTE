#ifndef DGSOLVER
#define DGSOLVER

#include <vector>
#include "Eigen/Dense"
#include "SpatialMesh/data_struct.h"

using namespace Eigen;
using namespace std;

vector<vector<vector<MatrixXd>>> CoefficientMat(vector<MatrixXd>& MassMat, vector<MatrixXd>& StfMatX, vector<MatrixXd>& StfMatY, \
                                        vector<vector<vector<vector<MatrixXd>>>>& MassOnFace_out, MatrixXd& Cx, MatrixXd& Cy, \
                                        vector<vector<vector<int>>>& CompOrder, int Np, int Na, int Nt, double Tau_C_inv);

vector<vector<MatrixXd>> DG_Solver_EDF(vector<vector<vector<int>>>& CompOrder, vector<Face>& Faces, vector<Cell>& Cells, \
                                        vector<MatrixXd>& MassMat, vector<vector<vector<vector<MatrixXd>>>>& MassOnFace_in, vector<vector<vector<vector<MatrixXd>>>>& FluxInt, \
                                        std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& IntOnFace, unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> FW, \
                                        vector<vector<vector<MatrixXd>>>& Asol, vector<vector<MatrixXd>>& EDF, vector<vector<MatrixXd>>& EDFn,\
                                        MatrixXd& Ts, MatrixXd& Qxs, MatrixXd& Qys, \
                                        MatrixXd& Cx, MatrixXd& Cy, VectorXd& BDtemp, MatrixXi& bdinf, \
                                        int deg, int DoF, int Nt, int Np, int Na, double Cv, double Vg, \
                                        double Tau_R_inv, double Tau_N_inv);
                    
#endif