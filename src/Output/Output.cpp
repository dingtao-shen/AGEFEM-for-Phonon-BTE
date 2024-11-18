#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include<cassert>
#include<vector>

#include "Eigen/Dense"
#include "Utility/Utility.h"
#include "SpatialMesh/data_struct.h"
#include "Output/Output.h"

using namespace std;
using namespace Eigen;

bool InDomain(double x, double y){
    if(x < -1.0 || x > 1.0 || y < -1.0 || y > 1.0){
        return false;
    }
    if(pow(x + 0.5, 2) + pow(y - 0.5, 2) < 0.01){
        return false;
    }
    if(pow(x + 0.5, 2) + pow(y + 0.5, 2) < 0.0625){
        return false;
    }
    if(pow(x - 0.5, 2) + pow(y, 2) < 0.01){
        return false;
    }
    if(pow(x - 0.5, 2) + pow(y - 1.0, 2) < 0.0625){
        return false;
    }
    if(pow(x - 0.5, 2) + pow(y + 1.0, 2) < 0.0625){
        return false;
    }
    return true;
}

void Output(int meshtype, std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Node> Nodes, std::vector<Face> Faces, std::vector<Cell> Cells, \
            Eigen::MatrixXd dOmega, Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, int N, double Cv, int deg, const std::string &Outputpath)
{
    int DoF = EDF[0][0].cols();
    int Nt = EDF[0][0].rows();
    int Np = dOmega.rows();
    int Na = dOmega.cols();
    MatrixXd stdRefBasis = StandardBasis(1, deg);

    MatrixXi CellID(N, N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            CellID(i, j) = -1;
        }
    }
    VectorXd Px(N), Py(N);
    for(int j = 0; j < N; j++){
        Px(j) = -1.0 + double(j) / double(N - 1) * 2.0;
    }
    Py = Px;

    VectorXd x(4), y(4);
    bool InCell;
    double area;
    if(meshtype == 1){ /* For NURBS-enhanced Mesh */
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N; i++){
                if(InDomain(Px(i), Py(j)) != true){
                    continue;
                }
                for(int k = 0; k < Nt; k++){
                    for(int n = 0; n < 3; n++){
                        x(n) = Nodes[Cells[k].vertexes[n]].x;
                        y(n) = Nodes[Cells[k].vertexes[n]].y;
                    }
                    x(3) = x(0);
                    y(3) = y(0);

                    InCell = true;
                    for(int il = 0; il < 3; il++){
                        area = 0.5 * (Px(i) * (y(il) - y(il+1)) - Py(j) * (x(il) - x(il+1)) + (x(il) * y(il+1) - x(il+1) * y(il)));
                        if (area < 0){
                            InCell = false;
                            break;
                        }
                    }
                    if(InCell){CellID(i, j) = k; }
                }
            }
        }
    }
    else if(meshtype == 0){ /* For Straight-sided Mesh */
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N; i++){
                for(int k = 0; k < Nt; k++){
                    for(int n = 0; n < 3; n++){
                        x(n) = Nodes[Cells[k].vertexes[n]].x;
                        y(n) = Nodes[Cells[k].vertexes[n]].y;
                    }
                    x(3) = x(0);
                    y(3) = y(0);

                    InCell = true;
                    for(int il = 0; il < 3; il++){
                        area = 0.5 * (Px(i) * (y(il) - y(il+1)) - Py(j) * (x(il) - x(il+1)) + (x(il) * y(il+1) - x(il+1) * y(il)));
                        if (area < 0){
                            InCell = false;
                        }
                    }
                    if(InCell){CellID(i, j) = k; }
                }
            }
        }
    }

    VectorXd Pm(DoF);
    MatrixXd T = MatrixXd::Zero(N, N);
    MatrixXd Qx = MatrixXd::Zero(N, N);
    MatrixXd Qy = MatrixXd::Zero(N, N);
    double  xi, eta, A, B, C, D, E, F, G, H, s, sx, sy;
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            if(CellID(i, j) == -1){
                continue;
            }
            else{
                if(Cells[CellID(i, j)].type == 0){
                    for(int n = 0; n < 3; n++){
                        x(n) = Nodes[Cells[CellID(i, j)].vertexes[n]].x;
                        y(n) = Nodes[Cells[CellID(i, j)].vertexes[n]].y;
                    }
                    x(3) = x(0);
                    y(3) = y(0);

                    A = (x(1) - x(0)) * (y(2) - y(0)) - (x(2) - x(0)) * (y(1) - y(0));
                    B = y(2) - y(0);
                    C = x(0) - x(2);
                    D = (x(2) - x(0)) * y(0) - (y(2) - y(0)) * x(0);
                    E = (x(2) - x(0)) * (y(1) - y(0)) - (x(1) - x(0)) * (y(2)-y(0));
                    F = y(1) - y(0);
                    G = x(0) - x(1);
                    H = (x(1) - x(0)) * y(0) - (y(1) - y(0)) * x(0);

                    xi = B / A * Px(i) + C / A * Py(j) + D / A;
                    eta = F / E * Px(i) + G / E * Py(j) + H / E;
                    
                    Pm.setZero();
                    for(int m = 0; m < DoF; m++){
                        int l = 0;
                        for (int kl = 0; kl <= deg; kl++){
                            for (int jl = 0; jl <= kl; jl++){
                                int il = kl - jl;
                                Pm(m) += stdRefBasis(m, l) * pow(xi, il) * pow(eta, jl);
                                l++;
                            }
                        }
                    }
                    s = 0.0;
                    sx = 0.0;
                    sy = 0.0;
                    for (int j2 = 0; j2 < Na; j2++){
                        for (int j1 = 0; j1 < Np; j1++){
                            for (int l = 0; l < DoF; l++){
                                s += dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                                sx += Cx(j1, j2) * dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                                sy += Cy(j1, j2) * dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                            } 
                        }
                    }

                    T(i, j) = s / Cv;
                    Qx(i, j) = sx;
                    Qy(i, j) = sy;
                }
                else if(Cells[CellID(i, j)].type == 1){
                    Pm.setZero();
                    for(int m = 0; m < DoF; m++){
                        int l = 0;
                        for (int kl = 0; kl <= deg; kl++){
                            for (int jl = 0; jl <= kl; jl++){
                                int il = kl - jl;
                                Pm(m) += Cells[CellID(i, j)].Basis(m, l) * pow(Px(i), il) * pow(Py(j), jl);
                                l++;
                            }
                        }
                    }
                    s = 0.0;
                    sx = 0.0;
                    sy = 0.0;
                    for (int j2 = 0; j2 < Na; j2++){
                        for (int j1 = 0; j1 < Np; j1++){
                            for (int l = 0; l < DoF; l++){
                                s += dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                                sx += Cx(j1, j2) * dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                                sy += Cy(j1, j2) * dOmega(j1, j2) * EDF[j1][j2](CellID(i, j), l) * Pm(l);
                            } 
                        }
                    }

                    T(i, j) = s / Cv;
                    Qx(i, j) = sx;
                    Qy(i, j) = sy;
                }
            }
        }
    }

    ofstream write_output(Outputpath);
    assert(write_output.is_open());
    write_output.setf(ios::fixed);
    write_output.precision(16);
    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            write_output << Px(i) << " " << Py(j) << " " \
            << T(i, j) << " " << Qx(i, j) << " " << Qy(i, j) << endl;
        }
    }
    write_output.close();
}

using namespace Eigen;
using namespace std;