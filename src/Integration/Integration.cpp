#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>
#include <vector>
#include <unordered_map>
#include "omp.h"

#include "Utility/Utility.h"
#include "SpatialMesh/data_struct.h"
#include "SpatialMesh/SpatialMesh.h"
#include "VelocityMesh/VelocityMesh.h"
#include "Integration/Integration.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

Integration::Integration(vector<Node>& Nodes, vector<Face>& Faces, vector<Cell>& Cells, Eigen::MatrixXd& dOmega, Eigen::MatrixXd& Cx, Eigen::MatrixXd& Cy, \
                    Eigen::MatrixXi& boundaryinf, int deg, int NPfc, int Nlmd, int Nthe)
{
    std::cout << ">>> Pre-Compute the integration..." << endl;

    MatrixXd stdRefBasis = StandardBasis(1, deg);
    int DoF3D = (deg + 1) * (deg + 2) / 2;
    int Nt = Cells.size();
    
    /* Compute the integration of the basis functions over the triangle */
    vector<double> IntEle(DoF3D, 0);
    IntMat.resize(Nt, DoF3D);
    IntMat.setZero();
    // Integrate the element x^a * y^b (a+b=d, d = 0, 1, ..., deg) in the basis function over the reference triangle
    int idx = 0;
    for (int d = 0; d <= deg; d++){
        for(int a = 0; a <= d; a ++){
            int b = d - a;
            IntEle[idx] = DivFactorial(1, a, b+1, d+2);
            idx++;
        }
    }
    // Multiply the coefficient to compute the integration of the basis function
    double s = 0.0;
    for (int i = 0; i < DoF3D; i++){
        s = 0.0;
        for (int j = 0; j < DoF3D; j++){
            s += stdRefBasis(i, j) * IntEle[j];
        }
        // Map the integration from the refrence triangle to the physical triangle
        for (int k = 0; k < Nt; k++){ 
            if(Cells[k].type == 0){
                IntMat(k, i) = s * 2.0 * Cells[k].area;
            } 
        }
    }

    for(int i = 0; i < Nt; i++){
        if(Cells[i].type == 0){
            continue;
        }
        else if(Cells[i].type == 1){
            IntMat.row(i).setZero();
            MatrixXd crvF = evl_Basis(Cells[i].Basis, Cells[i].QuadPts, deg); // Np x DoF
            VectorXd crvf;
            for(int m = 0; m < DoF3D; m++){
                crvf = crvF.col(m);
                IntMat(i, m) = QuadSum(crvf, Cells[i].QuadPts);
            }
        }
    }

    std::cout << "   >>> IntMat is ready." << endl;

    /* Compute the integration of the product of 2 basis functions over the triangle, a.k.a. the mass matrix */
    MatrixXd IntElement = MatrixXd::Zero(DoF3D, DoF3D);
    MassMat.resize(Nt, MatrixXd::Zero(DoF3D, DoF3D));
    // Integrate the element (x^a * y^b) * (x^s * y^t) = x^{a+s} * y^{b+t} (a+b = 0, 1, ... deg, s+t = 0, 1, ..., deg)
    // in the product of two basis functions over the reference triangle
    int idx1 = 0, idx2 = 0;
    for (int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            idx2 = 0;
            for(int d2 = 0; d2 <= deg; d2++){
                for(int s = 0; s <= d2; s++){
                    int t = d2 - s;
                    IntElement(idx1, idx2) = DivFactorial(1, a + s, b + t + 1, d1 + d2 + 2);
                    idx2++;
                }
            }
            idx1++;
        }
    }
    // Multiply the coefficient to compute the integration of the basis function
    s = 0.0;
    for(int i = 0; i < DoF3D; i++){
        for(int j = 0; j < DoF3D; j++){
            s = 0.0;
            for(int jj = 0; jj < DoF3D; jj++){
                for(int ii = 0; ii < DoF3D; ii++){
                    s += stdRefBasis(j ,jj) * stdRefBasis(i, ii) * IntElement(ii, jj);
                }
            }
            // Map the integration from the refrence triangle to the physical triangle
            for (int k = 0; k < Nt; k++){ 
                if(Cells[k].type == 0){
                    MassMat[k](i, j) = s * 2.0 * Cells[k].area;
                } 
            }
        }
    }
    for(int i = 0; i < Nt; i++){
        if(Cells[i].type == 0){
            continue;
        }
        else if(Cells[i].type == 1){
            MassMat[i] = MatrixXd::Zero(DoF3D, DoF3D);
            MatrixXd crvF = evl_Basis(Cells[i].Basis, Cells[i].QuadPts, deg); // Np x DoF
            for(int m = 0; m < DoF3D; m++){
                for(int n = 0; n < DoF3D; n++){
                    VectorXd crvfm = crvF.col(m);
                    VectorXd crvfn = crvF.col(n);
                    VectorXd crvf = crvfm.array() * crvfn.array();
                    MassMat[i](m, n) = QuadSum(crvf, Cells[i].QuadPts);
                }
            }
        }
    }

    std::cout << "   >>> MassMat is ready." << endl;

    /* Compute integration of product of one basis function and partial x/y derivative of one basis function over triangle */
    MatrixXd IntElementX = MatrixXd::Zero(DoF3D, DoF3D);
    MatrixXd IntElementY = MatrixXd::Zero(DoF3D, DoF3D);
    StfMatX.resize(Nt, MatrixXd::Zero(DoF3D, DoF3D));
    StfMatY.resize(Nt, MatrixXd::Zero(DoF3D, DoF3D));

    // Integrate the element a * (x^{a-1} * y^b) * (x^s * y^t) = a * x^{a-1+s} * y^{b+t} (a+b = 0, 1, ... deg, s+t = 0, 1, ..., deg)
    IntElement.setZero();
    idx1 = 0;
    idx2 = 0;
    for (int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if (b > 0)
            {
                idx2 = 0;
                for(int d2 = 0; d2 <= deg; d2++){
                    for(int s = 0; s <= d2; s++){
                        int t = d2 - s;
                        IntElement(idx1, idx2) = double(b) * DivFactorial(1, a + s, b + t, d1 - 1 + d2 + 2);
                        idx2++;
                    }
                }
            }
            idx1++;
        }
    }
    s = 0.0;
    for(int i = 0; i < DoF3D; i++){
        for(int j = 0; j < DoF3D; j++){
            s = 0.0;
            for(int jj = 0; jj < DoF3D; jj++){
                for(int ii = 0; ii < DoF3D; ii++){
                    s += stdRefBasis(j ,jj) * stdRefBasis(i, ii) * IntElement(ii, jj);
                }
            }
            IntElementX(i, j) = s;
        }
    }
    // Integrate the element b * (x^a * y^{b-1}) * (x^s * y^t) = b * x^{a+s} * y^{b-1+t} (a+b = 0, 1, ... deg, s+t = 0, 1, ..., deg)
    IntElement.setZero();
    idx1 = 0;
    idx2 = 0;
    for (int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if (a > 0)
            {
                idx2 = 0;
                for(int d2 = 0; d2 <= deg; d2++){
                    for(int s = 0; s <= d2; s++){
                        int t = d2 - s;
                        IntElement(idx1, idx2) = double(a) * DivFactorial(1, a + s - 1, b + t + 1, d1 - 1 + d2 + 2);
                        idx2++;
                    }
                }
            }
            idx1++;
        }
    }
    s = 0.0;
    for(int i = 0; i < DoF3D; i++){
        for(int j = 0; j < DoF3D; j++){
            s = 0.0;
            for(int jj = 0; jj < DoF3D; jj++){
                for(int ii = 0; ii < DoF3D; ii++){
                    s += stdRefBasis(j ,jj) * stdRefBasis(i, ii) * IntElement(ii, jj);
                }
            }
            IntElementY(i, j) = s;
        }
    }

    for(int i = 0; i < Nt; i++){
        if(Cells[i].type == 0){
            vector<double> x(3), y(3);
            for(int k = 0; k < 3; k++){
                x[k] = Nodes[Cells[i].vertexes[k]].x;
                y[k] = Nodes[Cells[i].vertexes[k]].y; 
            }
            s = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
            for(int j = 0; j < DoF3D; j++){
                for(int k = 0; k < DoF3D; k++){
                    StfMatX[i](j, k) = ((y[2] - y[0]) / s * IntElementX(j, k) + (y[0] - y[1]) / s * IntElementY(j, k)) * 2.0 * Cells[i].area;
                    StfMatY[i](j, k) = ((x[0] - x[2]) / s * IntElementX(j, k) + (x[1] - x[0]) / s * IntElementY(j, k)) * 2.0 * Cells[i].area;
                }
            }
        }
        else if(Cells[i].type == 1){
            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, Cells[i].QuadPts, deg); // Np x DoF
            MatrixXd dxLP = cal_dxBasis(Cells[i].Basis, deg);
            MatrixXd dyLP = cal_dyBasis(Cells[i].Basis, deg);
            MatrixXd dxLPvalue = evl_dBasis(dxLP, Cells[i].QuadPts, deg);
            MatrixXd dyLPvalue = evl_dBasis(dyLP, Cells[i].QuadPts, deg);
            for(int m = 0; m < DoF3D; m++){
                VectorXd crvf = LPvalue.col(m);
                for(int n = 0; n < DoF3D; n++){   
                    VectorXd crvfx = dxLPvalue.col(n);
                    VectorXd crvfy = dyLPvalue.col(n);
                    VectorXd crvFx = crvf.array() * crvfx.array();
                    VectorXd crvFy = crvf.array() * crvfy.array();
                    StfMatX[i](n, m) = QuadSum(crvFx, Cells[i].QuadPts);
                    StfMatY[i](n, m) = QuadSum(crvFy, Cells[i].QuadPts);
                }
            }
        }
    }

    std::cout << "   >>> StfMatX/StfMatY is ready." << endl;

    /* Compute the integration of the product of 2 basis functions over the edges of the triangle*/
    int Np = dOmega.rows();
    int Na = dOmega.cols();
    MassOnFace.resize(Np, vector<vector<vector<MatrixXd>>>(Na, vector<vector<MatrixXd>>(Nt, vector<MatrixXd>(3, MatrixXd::Zero(DoF3D, DoF3D)))));
    MassOnFace_in.resize(Np, vector<vector<vector<MatrixXd>>>(Na, vector<vector<MatrixXd>>(Nt, vector<MatrixXd>(3, MatrixXd::Zero(DoF3D, DoF3D)))));
    MassOnFace_out.resize(Np, vector<vector<vector<MatrixXd>>>(Na, vector<vector<MatrixXd>>(Nt, vector<MatrixXd>(3, MatrixXd::Zero(DoF3D, DoF3D)))));

    vector<MatrixXd> Int_Element_E;
    Int_Element_E.resize(3, MatrixXd::Zero(DoF3D, DoF3D));
    // 1st line: y=0 in the reference triangle 
    // x^b * y^a
    idx1 = 0;
    idx2 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if(a == 0){
                idx2 = 0;
                for(int d2 = 0; d2 <= deg; d2++){
                    for(int s = 0; s <= d2; s++){
                        int t = d2 - s;
                        if(s == 0){
                            Int_Element_E[0](idx1, idx2) = 1.0 / double(b + t + 1);
                        }
                        idx2++;
                    }
                }
            }
            idx1++;
        }
    }
    // 2nd line: y=1-x in the reference triangle
    idx1 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            idx2 = 0;
            for(int d2 = 0; d2 <= deg; d2++){
                for(int s = 0; s <= d2; s++){
                    int t = d2 - s;
                    Int_Element_E[1](idx1, idx2) = sqrt(2.0) * DivFactorial(1, a + s, b + t + 1, d1 + d2 + 1);
                    idx2++;
                }
            }
            idx1++;
        }
    }
    // 3rd line: x=0 in reference triangle
    idx1 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if(b == 0){
                int idx2 = 0;
                for(int d2 = 0; d2 <= deg; d2++){
                    for(int s = 0; s <= d2; s++){
                        int t = d2 - s;
                        if(t == 0){
                            Int_Element_E[2](idx1, idx2) = 1.0 / double(a + s + 1);
                        }
                        idx2++;
                    }
                }
            }
            idx1++;
        }
    }

    MatrixXd GL = GaussQuad(0.0, 1.0, NPfc);
    omp_set_num_threads(32);
    #pragma omp parallel for
    for(int e1 = 0; e1 < Np; e1++){
        for(int e2 = 0; e2 < Na; e2++){
            for(int i = 0; i < Nt; i++){
                if(Cells[i].type == 0){
                    vector<double> x(3), y(3);
                    for(int k = 0; k < 3; k++){
                        x[k] = Nodes[Cells[i].vertexes[k]].x;
                        y[k] = Nodes[Cells[i].vertexes[k]].y; 
                    }
                    VectorXd sv = VectorXd::Zero(3);
                    for(int j = 0; j < DoF3D; j++){
                        for(int k = 0; k < DoF3D; k++){
                            sv.setZero();
                            for(int jj = 0; jj < DoF3D; jj++){
                                for(int kk = 0; kk < DoF3D; kk++){
                                    for(int t = 0; t < 3; t++){
                                        sv(t) += stdRefBasis(j, jj) * stdRefBasis(k, kk) * Int_Element_E[t](jj, kk);
                                    }
                                }
                            }

                            double speed1 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 0) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 0);
                            double speed2 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 1) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 1);
                            double speed3 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 2);
                            MassOnFace_out[e1][e2][i][0](j, k) = sqrt(pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2)) * sv[0] * 0.5 * (speed1 + fabs(speed1));
                            MassOnFace_out[e1][e2][i][1](j, k) = sqrt(pow(x[2] - x[1], 2) + pow(y[2] - y[1], 2)) / sqrt(2.0) * sv[1] * 0.5 * (speed2 + fabs(speed2));
                            MassOnFace_out[e1][e2][i][2](j, k) = sqrt(pow(x[2] - x[0], 2) + pow(y[2] - y[0], 2)) * sv[2] * 0.5 * (speed3 + fabs(speed3));
                            MassOnFace_in[e1][e2][i][0](j, k) = sqrt(pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2)) * sv[0] * 0.5 * (speed1 - fabs(speed1));
                            MassOnFace_in[e1][e2][i][1](j, k) = sqrt(pow(x[2] - x[1], 2) + pow(y[2] - y[1], 2)) / sqrt(2.0) * sv[1] * 0.5 * (speed2 - fabs(speed2));
                            MassOnFace_in[e1][e2][i][2](j, k) = sqrt(pow(x[2] - x[0], 2) + pow(y[2] - y[0], 2)) * sv[2] * 0.5 * (speed3 - fabs(speed3));
                            MassOnFace[e1][e2][i][0](j, k) = sqrt(pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2)) * sv[0];
                            MassOnFace[e1][e2][i][1](j, k) = sqrt(pow(x[2] - x[1], 2) + pow(y[2] - y[1], 2)) / sqrt(2.0) * sv[1];
                            MassOnFace[e1][e2][i][2](j, k) = sqrt(pow(x[2] - x[0], 2) + pow(y[2] - y[0], 2)) * sv[2];
                        }
                    }
                }
                else if(Cells[i].type == 1){
                    vector<double> x(4), y(4);
                    for(int t = 0; t < 3; t++){
                        x[t] = Nodes[Cells[i].vertexes[t]].x;
                        y[t] = Nodes[Cells[i].vertexes[t]].y;
                    }
                    x[3] = x[0];
                    y[3] = y[0];
                    for(int t = 0; t < 3; t++){
                        if(Faces[Cells[i].faces[t]].type == 0) // straight-sided face
                        {
                            MatrixXd IntNodes = MatrixXd::Zero(2, GL.cols()); 
                            for(int m = 0; m < GL.cols(); m++){
                                IntNodes(0, m) = x[t] + GL(0, m) * (x[t+1] - x[t]);
                                IntNodes(1, m) = y[t] + GL(0, m) * (y[t+1] - y[t]);
                            }
                            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, IntNodes, deg);
                            double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, t) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, t);
                            double len = Faces[Cells[i].faces[t]].length;
                            for(int j = 0; j < DoF3D; j++){
                                for(int k = 0; k < DoF3D; k++){
                                    VectorXd crvf1 = LPvalue.col(j);
                                    VectorXd crvf2 = LPvalue.col(k);
                                    VectorXd crvF = crvf1.array() * crvf2.array();
                                    MassOnFace[e1][e2][i][t](j, k) = len * (crvF.dot(GL.row(1)));
                                    MassOnFace_in[e1][e2][i][t](j, k) = 0.5 * (speed - fabs(speed)) * len * (crvF.dot(GL.row(1)));
                                    MassOnFace_out[e1][e2][i][t](j, k) = 0.5 * (speed + fabs(speed)) * len * (crvF.dot(GL.row(1)));
                                }
                            }
                        }
                        else if(Faces[Cells[i].faces[t]].type == 1) // NURBS faces
                        {
                            MatrixXd IntNodes = Faces[Cells[i].faces[t]].QuadPts;
                            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, IntNodes, deg);
                            VectorXd Speed(IntNodes.cols());
                            Speed = Cx(e1, e2) * Cells[i].crv_outward_norm_vec.row(0).array() + Cy(e1, e2) * Cells[i].crv_outward_norm_vec.row(1).array();
                            for(int j = 0; j < DoF3D; j++){
                                for(int k = 0; k < DoF3D; k++){
                                    VectorXd crvf1 = LPvalue.col(j);
                                    VectorXd crvf2 = LPvalue.col(k);
                                    VectorXd crvF = crvf1.array() * crvf2.array();
                                    VectorXd crvFin = 0.5 * (Speed.array() - Speed.array().abs()) * crvf1.array() * crvf2.array();
                                    VectorXd crvFout = 0.5 * (Speed.array() + Speed.array().abs()) * crvf1.array() * crvf2.array();
                                    MassOnFace[e1][e2][i][t](j, k) = QuadSum(crvF, IntNodes);
                                    MassOnFace_in[e1][e2][i][t](j, k) = QuadSum(crvFin, IntNodes);
                                    MassOnFace_out[e1][e2][i][t](j, k) = QuadSum(crvFout, IntNodes);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "   >>> MassOnFace is ready." << endl;

    /* Computing integration of production of interior and exterior triangle basis functions over each edge */
    FluxInt.resize(Np, vector<vector<vector<MatrixXd>>>(Na, vector<vector<MatrixXd>>(Nt, vector<MatrixXd>(3, MatrixXd::Zero(DoF3D, DoF3D)))));

    unordered_map<int, vector<double>> BC_Trans;
    BC_Trans[2] = {0, -2};
    BC_Trans[7] = {2, 0};
    BC_Trans[8] = {0, -2};
    BC_Trans[4] = {0, 2};
    BC_Trans[3] = {-2, 0};
    BC_Trans[6] = {0, 2};

    omp_set_num_threads(32);
    #pragma omp parallel for
    for(int e1 = 0; e1 < Np; e1++){
        for(int e2 = 0; e2 < Na; e2++){
            for(int i = 0; i < Nt; i++){
                vector<double> x(4), y(4);
                for(int k = 0; k < 3; k++){
                    x[k] = Nodes[Cells[i].vertexes[k]].x;
                    y[k] = Nodes[Cells[i].vertexes[k]].y;
                }
                x[3] = x[0];
                y[3] = y[0];
                if(Cells[i].type == 0){
                    VectorXd pm(DoF3D), pl(DoF3D);

                    for(int j2 = 0; j2 < 3; j2++){
                        
                        int FCID = Cells[i].faces[j2];
                        int BCID = Faces[FCID].boundaryflag;
                        double len = Faces[FCID].length;
                        double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, j2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, j2);
                        int NBR = (Faces[FCID].neighbor_cell[0] == i) ? Faces[FCID].neighbor_cell[1] : Faces[FCID].neighbor_cell[0];

                        MatrixXd IntNodes = MatrixXd::Zero(2, GL.cols()); // integrals nodes on the straight face
                        for(int m = 0; m < GL.cols(); m++){
                            IntNodes(0, m) = x[j2] + GL(0, m) * (x[j2 + 1] - x[j2]);
                            IntNodes(1, m) = y[j2] + GL(0, m) * (y[j2 + 1] - y[j2]);
                        }

                        if(BCID == 0 && Cells[NBR].type == 0) // this faces is interior with a straight-sided neighbour cell
                        {
                            vector<double> nbr_x(4), nbr_y(4);
                            for(int k = 0; k < 3; k++){
                                nbr_x[k] = Nodes[Cells[NBR].vertexes[k]].x;
                                nbr_y[k] = Nodes[Cells[NBR].vertexes[k]].y;
                            }
                            nbr_x[3] = nbr_x[0];
                            nbr_y[3] = nbr_y[0];

                            for(int j1 = 0; j1 < GL.cols(); j1++){
                                // assert(len != 0);
                                double xp = IntNodes(0, j1);
                                double yp = IntNodes(1, j1);

                                double A = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
                                double B = y[2] - y[0];
                                double C = x[0] - x[2];
                                double D = (x[2] - x[0]) * y[0] - (y[2] - y[0]) * x[0];
                                double E = (x[2] - x[0]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2]-y[0]);
                                double F = y[1] - y[0];
                                double G = x[0] - x[1];
                                double H = (x[1] - x[0]) * y[0] - (y[1] - y[0]) * x[0];

                                double xi = B / A * xp + C / A * yp + D / A;
                                double eta = F / E * xp + G / E * yp + H / E;

                                pm.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b  = d - a;
                                            pm(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }

                                // neighbor triangle
                                A = (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]) - (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]);
                                B = nbr_y[2] - nbr_y[0];
                                C = nbr_x[0] - nbr_x[2];
                                D = (nbr_x[2] - nbr_x[0]) * nbr_y[0] - (nbr_y[2] - nbr_y[0]) * nbr_x[0];
                                E = (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]) - (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]);
                                F = nbr_y[1] - nbr_y[0];
                                G = nbr_x[0] - nbr_x[1];
                                H = (nbr_x[1] - nbr_x[0]) * nbr_y[0] - (nbr_y[1] - nbr_y[0]) * nbr_x[0];

                                xi = B / A * xp + C / A * yp + D / A;
                                eta = F / E * xp + G / E * yp + H / E;

                                pl.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b = d - a;
                                            pl(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }
                                
                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * pl(k) * len * GL(1, j1);
                                    }
                                }
                            }
                        }
                        else if(BCID == 0 && Cells[NBR].type == 1) // this faces is interior with a NURBS-enhanced neighbour cell
                        {
                            MatrixXd LPvalue = evl_Basis(Cells[NBR].Basis, IntNodes, deg);
                            double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, j2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, j2);
                            for(int j1 = 0; j1 < GL.cols(); j1++){
                                // assert(len != 0);
                                double xp = IntNodes(0, j1);
                                double yp = IntNodes(1, j1);

                                double A = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
                                double B = y[2] - y[0];
                                double C = x[0] - x[2];
                                double D = (x[2] - x[0]) * y[0] - (y[2] - y[0]) * x[0];
                                double E = (x[2] - x[0]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2]-y[0]);
                                double F = y[1] - y[0];
                                double G = x[0] - x[1];
                                double H = (x[1] - x[0]) * y[0] - (y[1] - y[0]) * x[0];

                                double xi = B / A * xp + C / A * yp + D / A;
                                double eta = F / E * xp + G / E * yp + H / E;

                                pm.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b  = d - a;
                                            pm(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }

                                pl = LPvalue.row(j1);
                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * pl(k) * len * GL(1, j1);
                                    }
                                }
                            }
                        }
                        else if(boundaryinf(2, BCID - 1) == 3){
                            vector<double> nbr_x(4), nbr_y(4);
                            for(int k = 0; k < 3; k++){
                                nbr_x[k] = Nodes[Cells[NBR].vertexes[k]].x;
                                nbr_y[k] = Nodes[Cells[NBR].vertexes[k]].y;
                            }
                            nbr_x[3] = nbr_x[0];
                            nbr_y[3] = nbr_y[0];
                            double tx = BC_Trans[BCID][0];
                            double ty = BC_Trans[BCID][1];
                            for(int j1 = 0; j1 < GL.cols(); j1++){
                                // assert(len != 0);
                                double xp = IntNodes(0, j1);
                                double yp = IntNodes(1, j1);
                                double xp_nbr = xp - tx;
                                double yp_nbr = yp - ty;

                                double A = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
                                double B = y[2] - y[0];
                                double C = x[0] - x[2];
                                double D = (x[2] - x[0]) * y[0] - (y[2] - y[0]) * x[0];
                                double E = (x[2] - x[0]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2]-y[0]);
                                double F = y[1] - y[0];
                                double G = x[0] - x[1];
                                double H = (x[1] - x[0]) * y[0] - (y[1] - y[0]) * x[0];

                                double xi = B / A * xp + C / A * yp + D / A;
                                double eta = F / E * xp + G / E * yp + H / E;

                                pm.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b  = d - a;
                                            pm(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }

                                // neighbor triangle
                                A = (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]) - (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]);
                                B = nbr_y[2] - nbr_y[0];
                                C = nbr_x[0] - nbr_x[2];
                                D = (nbr_x[2] - nbr_x[0]) * nbr_y[0] - (nbr_y[2] - nbr_y[0]) * nbr_x[0];
                                E = (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]) - (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]);
                                F = nbr_y[1] - nbr_y[0];
                                G = nbr_x[0] - nbr_x[1];
                                H = (nbr_x[1] - nbr_x[0]) * nbr_y[0] - (nbr_y[1] - nbr_y[0]) * nbr_x[0];

                                xi = B / A * xp_nbr + C / A * yp_nbr + D / A;
                                eta = F / E * xp_nbr + G / E * yp_nbr + H / E;

                                pl.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b = d - a;
                                            pl(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }
                                
                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * pl(k) * len * GL(1, j1);
                                    }
                                }
                            }
                        }
                        else if(boundaryinf(2, BCID - 1) == 2){
                            continue; // no flux integration is need for non-thermalizing boundary
                        }
                        else if(boundaryinf(2, BCID - 1) == 1)
                        {
                            assert(boundaryinf(2, BCID-1) == 1); // this situation is not involved in this mesh
                            double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, j2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, j2);
                            for(int j1 = 0; j1 < IntNodes.cols(); j1++){
                                double xp = IntNodes(0, j1);
                                double yp = IntNodes(1, j1);

                                double A = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
                                double B = y[2] - y[0];
                                double C = x[0] - x[2];
                                double D = (x[2] - x[0]) * y[0] - (y[2] - y[0]) * x[0];
                                double E = (x[2] - x[0]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2]-y[0]);
                                double F = y[1] - y[0];
                                double G = x[0] - x[1];
                                double H = (x[1] - x[0]) * y[0] - (y[1] - y[0]) * x[0];

                                double xi = B / A * xp + C / A * yp + D / A;
                                double eta = F / E * xp + G / E * yp + H / E;

                                pm.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b  = d - a;
                                            pm(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }

                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * len * GL(1, j1);
                                    }
                                }
                            }
                        }
                    }
                }
                else if(Cells[i].type == 1){
                    VectorXd pm(DoF3D), pl(DoF3D);
                    for(int j2 = 0; j2 < 3; j2++){
                        int FCID = Cells[i].faces[j2];
                        int NBR = (Faces[FCID].neighbor_cell[0] == i) ? Faces[FCID].neighbor_cell[1] : Faces[FCID].neighbor_cell[0];
                        int BCID = Faces[FCID].boundaryflag;
                        double len = Faces[FCID].length;
                        if(BCID == 0 && Cells[NBR].type == 0) // this is the interior straight-sided face, with a straight-sided neighbour cell
                        {
                            // int NBR = (Faces[FCID].neighbor_cell[0] == i) ? Faces[FCID].neighbor_cell[1] : Faces[FCID].neighbor_cell[0];
                            vector<double> nbr_x(4), nbr_y(4);
                            for(int k = 0; k < 3; k++){
                                nbr_x[k] = Nodes[Cells[NBR].vertexes[k]].x;
                                nbr_y[k] = Nodes[Cells[NBR].vertexes[k]].y;
                            }
                            nbr_x[3] = nbr_x[0];
                            nbr_y[3] = nbr_y[0];
                            MatrixXd IntNodes = MatrixXd::Zero(2, GL.cols()); // integrals nodes on the straight face
                            for(int m = 0; m < GL.cols(); m++){
                                IntNodes(0, m) = x[j2] + GL(0, m) * (x[j2 + 1] - x[j2]);
                                IntNodes(1, m) = y[j2] + GL(0, m) * (y[j2 + 1] - y[j2]);
                            }
                            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, IntNodes, deg);
                            double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, j2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, j2);
                            for(int j1 = 0; j1 < GL.cols(); j1++){
                                // assert(len != 0);
                                double xp = IntNodes(0, j1);
                                double yp = IntNodes(1, j1);

                                double A = (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]) - (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]);
                                double B = nbr_y[2] - nbr_y[0];
                                double C = nbr_x[0] - nbr_x[2];
                                double D = (nbr_x[2] - nbr_x[0]) * nbr_y[0] - (nbr_y[2] - nbr_y[0]) * nbr_x[0];
                                double E = (nbr_x[2] - nbr_x[0]) * (nbr_y[1] - nbr_y[0]) - (nbr_x[1] - nbr_x[0]) * (nbr_y[2] - nbr_y[0]);
                                double F = nbr_y[1] - nbr_y[0];
                                double G = nbr_x[0] - nbr_x[1];
                                double H = (nbr_x[1] - nbr_x[0]) * nbr_y[0] - (nbr_y[1] - nbr_y[0]) * nbr_x[0];

                                double xi = B / A * xp + C / A * yp + D / A;
                                double eta = F / E * xp + G / E * yp + H / E;

                                pl.setZero();
                                for(int idx1 = 0; idx1 < DoF3D; idx1++){
                                    int idx2 = 0;
                                    for(int d = 0; d <= deg; d++){
                                        for(int a = 0; a <= d; a++){
                                            int b  = d - a;
                                            pl(idx1) += stdRefBasis(idx1, idx2) * pow(xi, b) * pow(eta, a);
                                            idx2++;
                                        }
                                    }
                                }

                                pm = LPvalue.row(j1);
                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * pl(k) * len * GL(1, j1);
                                    }
                                }
                            }
                            
                        }
                        else if(BCID == 0 && Cells[NBR].type == 1){ // this is the interior straight-sided face, with a NURBS-enhanced neighbour cell
                            MatrixXd IntNodes = MatrixXd::Zero(2, GL.cols()); // integrals nodes on the straight face
                            for(int m = 0; m < GL.cols(); m++){
                                IntNodes(0, m) = x[j2] + GL(0, m) * (x[j2 + 1] - x[j2]);
                                IntNodes(1, m) = y[j2] + GL(0, m) * (y[j2 + 1] - y[j2]);
                            }
                            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, IntNodes, deg);
                            MatrixXd LPvalue_nbr = evl_Basis(Cells[NBR].Basis, IntNodes, deg);
                            double speed = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, j2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, j2);
                            for(int j1 = 0; j1 < GL.cols(); j1++){
                                pl = LPvalue_nbr.row(j1);
                                pm = LPvalue.row(j1);
                                for(int j = 0; j < DoF3D; j++){
                                    for(int k = 0; k < DoF3D; k++){
                                        FluxInt[e1][e2][i][j2](j, k) += 0.5 * (speed - fabs(speed)) * pm(j) * pl(k) * len * GL(1, j1);
                                    }
                                }
                            }
                        }
                        else{ // here we assume the curved boundaries are only equipped with diffuse or isothermalizing boundary condition.
                            /* In the context of new boundary condition, all the curved boundaries are equipped wit non-thermalizing condition */
                            MatrixXd IntNodes = Faces[FCID].QuadPts;
                            MatrixXd LPvalue = evl_Basis(Cells[i].Basis, IntNodes, deg);
                            VectorXd Speed(IntNodes.cols());
                            Speed = Cx(e1, e2) * Cells[i].crv_outward_norm_vec.row(0).array() + Cy(e1, e2) * Cells[i].crv_outward_norm_vec.row(1).array();
                            for(int j = 0; j < DoF3D; j++){
                                VectorXd crvf = 0.5 * (Speed.array() - Speed.array().abs()) * LPvalue.col(j).array();
                                for(int k = 0; k < DoF3D; k++){
                                    FluxInt[e1][e2][i][j2](j, k) = QuadSum(crvf, IntNodes);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "   >>> FluxInt is ready." << endl;

    /* Compute the integration of the basis function over the straight-sided face */
    MatrixXd Int_Element_e = MatrixXd::Zero(3, DoF3D);
    // 1st line: y=0 in the reference triangle 
    // x^b * y^a
    idx1 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if(a == 0){
                Int_Element_e(0, idx1) = 1.0 / double(b + 1);
            }
            idx1++;
        }
    }
    // 2nd line: y=1-x in the reference triangle
    idx1 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            Int_Element_e(1, idx1) = sqrt(2.0) * DivFactorial(1, a, b + 1, deg + 1);
            idx1++;
        }
    }
    // 3rd line: x=0 in reference triangle
    idx1 = 0;
    for(int d1 = 0; d1 <= deg; d1++){
        for(int a = 0; a <= d1; a++){
            int b = d1 - a;
            if(b == 0){
                Int_Element_e(2, idx1) = 1.0 / double(a + 1);
            }
            idx1++;
        }
    }

    IntOnFace.resize(Np, vector<vector<MatrixXd>>(Na, vector<MatrixXd>(Nt, MatrixXd::Zero(3, DoF3D))));
    omp_set_num_threads(32);
    #pragma omp parallel for
    for(int e1 = 0; e1 < Np; e1++){
        for(int e2 = 0; e2 < Na; e2++){
            for(int i = 0; i < Nt; i++){
                vector<double> x(3), y(3);
                for(int k = 0; k < 3; k++){
                    x[k] = Nodes[Cells[i].vertexes[k]].x;
                    y[k] = Nodes[Cells[i].vertexes[k]].y; 
                }
                VectorXd sv = VectorXd::Zero(3);
                for(int j = 0; j < DoF3D; j++){
                    sv.setZero();
                    for(int jj = 0; jj < DoF3D; jj++){
                        for(int t = 0; t < 3; t++){
                            sv(t) += stdRefBasis(j, jj) * Int_Element_e(t, jj);
                        }
                    }
                    double speed1 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 0) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 0);
                    double speed2 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 1) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 1);
                    double speed3 = Cx(e1, e2) * Cells[i].str_outward_norm_vec(0, 2) + Cy(e1, e2) * Cells[i].str_outward_norm_vec(1, 2);
                    IntOnFace[e1][e2][i](0, j) = sqrt(pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2)) * sv[0] * 0.5 * (speed1 - fabs(speed1));
                    IntOnFace[e1][e2][i](1, j) = sqrt(pow(x[2] - x[1], 2) + pow(y[2] - y[1], 2)) / sqrt(2.0) * sv[1] * 0.5 * (speed2 - fabs(speed2));
                    IntOnFace[e1][e2][i](2, j) = sqrt(pow(x[2] - x[0], 2) + pow(y[2] - y[0], 2)) * sv[2] * 0.5 * (speed3 - fabs(speed3));
                }
            }
        }
    }

    std::cout << ">>> Pre-Computation of the integration is done." << endl;
}

Integration::~Integration(){}

Eigen::MatrixXd Integration::GetIntMat(){ return IntMat; }
std::vector<Eigen::MatrixXd> Integration::GetMassMat(){ return MassMat; }
std::vector<Eigen::MatrixXd> Integration::GetStfMatX(){ return StfMatX; }
std::vector<Eigen::MatrixXd> Integration::GetStfMatY(){ return StfMatY; }
std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> Integration::GetMassOnFace(){ return MassOnFace; }
std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> Integration::GetMassOnFace_in(){ return MassOnFace_in; }
std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> Integration::GetMassOnFace_out(){ return MassOnFace_out; }
std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> Integration::GetFluxInt(){ return FluxInt; }
std::vector<std::vector<std::vector<Eigen::MatrixXd>>> Integration::GetIntOnFace(){ return IntOnFace; }