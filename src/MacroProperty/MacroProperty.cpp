#include<iostream>
#include<fstream>
#include<math.h>
#include<cassert>
#include"omp.h"
#include <vector>
#include <unordered_map>
#include "Eigen/Dense"
#include "Utility/Utility.h"
#include "MacroProperty/MacroProperty.h"

using namespace std;
using namespace Eigen;

MacroProperty::MacroProperty(int Nt, int DoF){
    T.resize(Nt);
    T.setZero();

    Qx.resize(Nt);
    Qx.setZero();

    Qy.resize(Nt);
    Qy.setZero();

    T_old.resize(Nt);
    T_old.setZero();

    Qx_old.resize(Nt);
    Qx_old.setZero();

    Qy_old.resize(Nt);
    Qy_old.setZero();

    Ts.resize(Nt, DoF);
    Ts.setZero();

    Qxs.resize(Nt, DoF);
    Qxs.setZero();

    Qys.resize(Nt, DoF);
    Qys.setZero();

    ResidualT = 1.0;
}

MacroProperty::~MacroProperty(){}

void MacroProperty::Cal_MacroProperty(std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Face> Faces, std::vector<Cell> Cells, \
                            Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, Eigen::MatrixXd dOmega, Eigen::MatrixXd IntMat, int Nt, int DoF, \
                            double Cv)
{
    T.setZero();
    Qx.setZero();
    Qy.setZero();
    Ts.setZero();
    Qxs.setZero();
    Qys.setZero();

    for(int i = 0; i < Cx.rows(); i++){
        for(int j = 0; j < Cx.cols(); j++){
            Ts = Ts.array() + dOmega(i, j) / Cv * EDF[i][j].array();
            Qxs = Qxs.array() + Cx(i, j) * dOmega(i, j) * EDF[i][j].array();
            Qys = Qys.array() + Cy(i, j) * dOmega(i, j) * EDF[i][j].array();
        }
    }

    T = ( Ts.array() * IntMat.array() ).rowwise().sum();
    Qx = ( Qxs.array() * IntMat.array() ).rowwise().sum();
    Qy = ( Qys.array() * IntMat.array() ).rowwise().sum();
}

double MacroProperty::Cal_ResidualT(){
    assert(T.norm() > 1.0e-18);
    ResidualT = (T - T_old).norm() / T.norm();
    assert(!isnan(ResidualT));
    T_old = T;
    return ResidualT;
}

VectorXd MacroProperty::GetT(){ return T; }
VectorXd MacroProperty::GetT_old(){ return T_old; }
MatrixXd MacroProperty::GetTs(){ return Ts; }
MatrixXd MacroProperty::GetQxs(){ return Qxs; }
MatrixXd MacroProperty::GetQys(){ return Qys; }

unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> Cal_Flux_Wall(std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Face> Faces, std::vector<Cell> Cells, \
                            Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, Eigen::MatrixXd dOmega, Eigen::MatrixXi& bdinf, \
                            std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>>& MassOnFace, \
                            int Np, int Na, int Nf, int DoF3D, int deg)
{
    unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> FW;
    for(int i = 0; i < Nf; i++){
        if(Faces[i].boundaryflag > 0 && bdinf(2, Faces[i].boundaryflag - 1) == 2){
            FW[i] = vector<vector<VectorXd>>(Np, vector<VectorXd>(Na, VectorXd::Zero(DoF3D)));
            int TRID = (Faces[i].neighbor_cell[0] != -1) ? Faces[i].neighbor_cell[0] : Faces[i].neighbor_cell[1];
            int LFID = (Faces[i].local_indices[0] != -1) ? Faces[i].local_indices[0] : Faces[i].local_indices[1];
            if(Faces[i].type == 0){
                double n1 = Cells[TRID].str_outward_norm_vec(0, LFID);
                double n2 = Cells[TRID].str_outward_norm_vec(1, LFID);

                omp_set_num_threads(32);
                #pragma omp parallel for
                for(int e2 = 0; e2 < Na; e2++){
                    for(int e1 = 0; e1 < Np; e1++){
                        double speed = Cx(e1, e2) * n1 + Cy(e1, e2) * n2;
                        if(speed < 0.0){
                            for(int m = 0; m < DoF3D; m++){
                                double s = 0.0;
                                for(int j2 = 0; j2 < Na; j2++){
                                    for(int j1 = 0; j1 < Np; j1++){
                                        double u = Cx(j1, j2) * n1 + Cy(j1, j2) * n2;
                                        if(u > 0.0){
                                            for(int l = 0; l < DoF3D; l++){
                                                s += u * dOmega(j1, j2) * EDF[j1][j2](TRID, l) * MassOnFace[j1][j2][TRID][LFID](m, l);
                                            }
                                        }
                                    }
                                }
                                FW[i][e1][e2](m) = speed * s / M_PI;
                            }
                        }
                    }
                }
            }
            else if(Faces[i].type == 1){
                MatrixXd IntNodes = Faces[i].QuadPts;
                MatrixXd LPvalue = evl_Basis(Cells[TRID].Basis, IntNodes, deg);

                omp_set_num_threads(32);
                #pragma omp parallel for
                for(int e2 = 0; e2 < Na; e2++){
                    for(int e1 = 0; e1 < Np; e1++){
                        VectorXd Speed = Cx(e1, e2) * Cells[TRID].crv_outward_norm_vec.row(0).array() + Cy(e1, e2) * Cells[TRID].crv_outward_norm_vec.row(1).array();
                        VectorXd speed;
                        for(int m = 0; m < DoF3D; m++){
                            VectorXd E = VectorXd::Zero(IntNodes.cols());
                            for(int j = 0; j < DoF3D; j++){
                                VectorXd e = VectorXd::Zero(IntNodes.cols());
                                for(int j2 = 0; j2 < Na; j2++){
                                    for(int j1 = 0; j1 < Np; j1++){
                                        speed = Cx(j1, j2) * Cells[TRID].crv_outward_norm_vec.row(0).array() + Cy(j1, j2) * Cells[TRID].crv_outward_norm_vec.row(1).array();
                                        e = e.array() + (speed.array() + speed.array().abs()).array() * speed.array().abs() * dOmega(j1, j2) * EDF[j1][j2](TRID, j);
                                    }
                                }
                                E = E.array() + 0.25 / M_PI * (Speed.array() - Speed.array().abs()) * e.array() * LPvalue.col(j).array();
                            }
                            VectorXd f = E.array() * LPvalue.col(m).array();
                            FW[i][e1][e2](m) = QuadSum(f, IntNodes);
                        }
                    }
                }
            }
        }
    }
    return FW;
}