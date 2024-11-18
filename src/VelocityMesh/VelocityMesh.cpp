#include<iostream>
#include<math.h>
#include<cassert>

#include "VelocityMesh/VelocityMesh.h"
#include "Utility/Utility.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

VelocityMesh::VelocityMesh(int Np, int Na, double Vg, double Cv){
    cout << ">>> Initialize the velocity mesh ..." << endl;

    dOmega.resize(Np, Na);
    dOmega.setZero();
    Cx.resize(Np, Na);
    Cx.setZero();
    Cy.resize(Np, Na);
    Cy.setZero();

    MatrixXd GQ_the = GaussQuad(0, M_PI, Np);
    VectorXd Theta = GQ_the.row(0).reverse();
    VectorXd wTheta = GQ_the.row(1).reverse();

    VectorXd Phi = VectorXd::Zero(Na);
    VectorXd wPhi = VectorXd::Zero(Na);
    MatrixXd GQ_phi = GaussQuad(0, M_PI, Na / 2);
    Phi.segment(0, Na / 2) = GQ_phi.row(0).reverse();
    wPhi.segment(0, Na / 2) = GQ_phi.row(1).reverse();
    GQ_phi = GaussQuad(M_PI, 2.0 * M_PI, Na / 2);
    Phi.segment(Na / 2, Na / 2) = GQ_phi.row(0).reverse();
    wPhi.segment(Na / 2, Na / 2) = GQ_phi.row(1).reverse();

    for(int i = 0; i < Np; i++){
        for(int j = 0; j < Na; j++){
            // dOmega(i, j) = sin(Theta(i)) * wTheta(i) * wPhi(j);
            // Cx(i, j) = Vg * sin(Theta(i)) * cos(Phi(j));
            // Cy(i, j) = Vg * sin(Theta(i)) * sin(Phi(j));
            dOmega(i, j) = wPhi(j);
            Cx(i, j) = Vg * cos(Phi(j));
            Cy(i, j) = Vg * sin(Phi(j));
        }
    }

    cout << ">>> The velocity mesh is ready!" << endl;
};

VelocityMesh::~VelocityMesh(){};

MatrixXd VelocityMesh::GetdOmega(){
    return dOmega;
};

MatrixXd VelocityMesh::GetCx(){
    return Cx;
};

MatrixXd VelocityMesh::GetCy(){
    return Cy;
};