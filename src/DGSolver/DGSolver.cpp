#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include<cassert>
#include <unordered_map>
#include "omp.h"

#include "Eigen/Dense"
#include "Utility/Utility.h"
#include "SpatialMesh/data_struct.h"
#include "SpatialMesh/SpatialMesh.h"
#include "VelocityMesh/VelocityMesh.h"
#include "Integration/Integration.h"
#include "MacroProperty/MacroProperty.h"
#include "DGSolver/DGSolver.h"

using namespace std;
using namespace Eigen;

vector<vector<vector<MatrixXd>>> CoefficientMat(vector<MatrixXd>& MassMat, vector<MatrixXd>& StfMatX, vector<MatrixXd>& StfMatY, \
                                        vector<vector<vector<vector<MatrixXd>>>>& MassOnFace_out, MatrixXd& Cx, MatrixXd& Cy, \
                                        vector<vector<vector<int>>>& CompOrder, int Np, int Na, int Nt, double Tau_C_inv)
{
    int DoF = MassMat[0].rows();
    vector<vector<vector<MatrixXd>>> Asol;
    Asol.resize(Np, vector<vector<MatrixXd>>(Na, vector<MatrixXd>(Nt, MatrixXd::Zero(DoF, DoF))));
    omp_set_num_threads(32);
    #pragma omp parallel for
    for(int j2 = 0; j2 < Na; j2++){
        for(int j1 = 0; j1 < Np; j1++){
            for(int i = 0; i < Nt; i++){
                int CellID = CompOrder[j1][j2][i];
                Asol[j1][j2][CellID] = Tau_C_inv * MassMat[CellID].array() - Cx(j1, j2) * StfMatX[CellID].array() - Cy(j1, j2) * StfMatY[CellID].array() \
                        + MassOnFace_out[j1][j2][CellID][0].array() \
                        + MassOnFace_out[j1][j2][CellID][1].array() \
                        + MassOnFace_out[j1][j2][CellID][2].array();
            }
        }
    }
    return Asol;
}

vector<vector<MatrixXd>> DG_Solver_EDF(vector<vector<vector<int>>>& CompOrder, vector<Face>& Faces, vector<Cell>& Cells, \
                                        vector<MatrixXd>& MassMat, vector<vector<vector<vector<MatrixXd>>>>& MassOnFace_in, vector<vector<vector<vector<MatrixXd>>>>& FluxInt, \
                                        std::vector<std::vector<std::vector<Eigen::MatrixXd>>>& IntOnFace, unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> FW,\
                                        vector<vector<vector<MatrixXd>>>& Asol, vector<vector<MatrixXd>>& EDF, vector<vector<MatrixXd>>& EDFn, \
                                        MatrixXd& Ts, MatrixXd& Qxs, MatrixXd& Qys, \
                                        MatrixXd& Cx, MatrixXd& Cy, VectorXd& BDtemp, MatrixXi& bdinf, \
                                        int deg, int DoF, int Nt, int Np, int Na, double Cv, double Vg, \
                                        double Tau_R_inv, double Tau_N_inv)
{
    omp_set_num_threads(32);
    #pragma omp parallel for
    for(int j2 = 0; j2 < Na; j2 ++){
        for(int j1 = 0; j1 < Np; j1++){
            for(int i = 0; i < Nt; i++){

                int CellID = CompOrder[j1][j2][i];

                VectorXd A_src = VectorXd::Zero(DoF);
                for(int l = 0; l < DoF; l++){
                    VectorXd vr = Cv * Ts(CellID, l) / 2.0 / M_PI * Tau_R_inv * MassMat[CellID].row(l).array();
                    VectorXd vn = Tau_N_inv * (Cv * Ts(CellID, l) / 2.0 / M_PI + 2 / 2.0 / M_PI * (Qxs(CellID, l) * Cx(j1, j2) + Qys(CellID, l) * Cy(j1, j2)) / pow(Vg, 2)) \
                                    * MassMat[CellID].row(l).array();
                    A_src = A_src + vr + vn;
                }

                for(int il = 0; il < 3; il++){
                    int FCID = Cells[CellID].faces[il];
                    int BCID = Faces[FCID].boundaryflag; 
                    if(BCID == 0){ // inner face
                        int NBR = (Faces[FCID].neighbor_cell[0] == CellID) ? Faces[FCID].neighbor_cell[1] : Faces[FCID].neighbor_cell[0];
                        for(int l = 0; l < DoF; l++){
                            A_src = A_src.array() - FluxInt[j1][j2][CellID][il].col(l).array() * EDF[j1][j2](NBR, l);
                        }
                    }
                    else if(bdinf(2, BCID - 1) == 1){ // thermalizing boundary condition
                        A_src = A_src.array() - Cv / 2.0 / M_PI * BDtemp(BCID - 1) * FluxInt[j1][j2][CellID][il].col(0).array();
                    }
                    else if(bdinf(2, BCID - 1) == 2){ // diffusely reflecting boundary condition
                        A_src = A_src.array() - FW[FCID][j1][j2].array();
                        // if(Faces[FCID].type == 0){
                        //     A_src = A_src.array() - FW[FCID][j1][j2].array();
                        // }
                        // else if(Faces[FCID].type == 1){
                        //     for(int l = 0; l < DoF; l++){
                        //         A_src = A_src.array() - FW[FCID][j1][j2](l) * MassOnFace_in[j1][j2][CellID][il].col(l).array();
                        //     }
                        // }
                    }
                    else if(bdinf(2, BCID - 1) == 3){  // periodic boudnary condition
                        double dT = (BCID == 8 || BCID == 2) ? -1.0 : 1.0;
                        int NBR = (Faces[FCID].neighbor_cell[0] == CellID) ? Faces[FCID].neighbor_cell[1] : Faces[FCID].neighbor_cell[0];
                        for(int l = 0; l < DoF; l++){
                            A_src = A_src.array() - FluxInt[j1][j2][CellID][il].col(l).array() * EDF[j1][j2](NBR, l);
                        }
                        A_src = A_src.array() - dT * Cv / (2 * M_PI) * IntOnFace[j1][j2][CellID].row(il).transpose().array();
                    }
                    else if(bdinf(2, BCID - 1) == 4){ // specularly reflecting boundary condition
                        for(int l = 0; l < DoF; l++){
                            int j2_sr = (j2 < Na / 2) ? Na / 2 - j2 - 1 : 3 * Na / 2 - j2 - 1; // find the incident direction
                            A_src = A_src.array() - EDFn[j1][j2_sr](CellID, l) * MassOnFace_in[j1][j2][CellID][il].col(l).array();
                        }
                    }
                }
                
                assert(!isnan(A_src.norm()));
                A_src = Asol[j1][j2][CellID].lu().solve(A_src);
                assert(!isnan(A_src.norm()));
                EDF[j1][j2].row(CellID) = A_src;
            }
        }
    }
    return EDF;
}