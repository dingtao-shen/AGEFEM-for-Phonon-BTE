#include <iostream>
#include <math.h>
#include <vector>

#include "Utility/Utility.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

MatrixXd GaussQuad(double a, double b, int N){
    double alpha = 0.0, beta = 0.0;
    MatrixXd GQ(2, N);
    if(N == 1){
        GQ(0, 0) = (alpha - beta) / (alpha + beta +2);
        GQ(1, 0) = 2;
        return GQ;
    }
    // Form symmetric matrix from recurrence.
    MatrixXd J = MatrixXd::Zero(N, N);
    VectorXd h = VectorXd::LinSpaced(N, 0, N-1);
    h = 2.0 * h.array() + alpha + beta;
    if(alpha + beta < 1e-15){
        J(0, 0) = 0.0;
    }
    else{
        J(0, 0) = -0.5 * (pow(alpha, 2) - pow(beta, 2)) / (h(0)+2.0) / h(0);
    }
    for(int i = 1; i < N; i++){
        J(i, i) = -0.5 * (pow(alpha, 2) - pow(beta, 2)) / (h(i)+2.0) / h(i);
        J(i-1, i) = 2.0 / (h(i-1)+2.0) * sqrt(i * (i + alpha + beta) * (i + alpha) * (i + beta) / (h(i-1) + 1.0) / (h(i-1) + 3.0));
    }
    MatrixXd J_ = J.transpose();
    J = J.array() + J_.array();
    EigenSolver<MatrixXd> eig(J);
    GQ.row(0) = eig.eigenvalues().real();
    GQ.row(1) = eig.eigenvectors().real().row(0).array().pow(2) * pow(2.0, alpha + beta + 1) / (alpha + beta + 1) 
                * tgamma(alpha + 1) * tgamma(beta + 1) / tgamma(alpha + beta + 1);

    MatrixXd GQ_sorted(2, N);
    VectorXd vec = GQ.row(0);
    VectorXi ind = VectorXi::LinSpaced(vec.size(), 0, vec.size()-1);
    auto rule = [vec](int i, int j) -> bool{return vec(i) < vec(j);};
    std::sort(ind.data(), ind.data()+ind.size(), rule);
    for(int i = 0; i < vec.size(); i++){
        GQ_sorted.col(i) = GQ.col(ind(i));
    }

    VectorXd x = GQ_sorted.row(0);
    VectorXd w = GQ_sorted.row(1);
    GQ_sorted.row(0) = 0.5 * ((b-a) * x.array() + (a+b)) ;
    GQ_sorted.row(1) = 0.5 * (b-a) * w.array();

    return GQ_sorted;
}

MatrixXd CrvQuadPts(double r, double start, double end, int n_gl){
    assert(start < end);
    double dir = (r == 0.1) ? -1.0 : 1.0;
    MatrixXd gl = GaussQuad(start, end, n_gl);

    MatrixXd Qpw = MatrixXd::Zero(5, gl.cols());
    for (int j = 0; j < gl.cols(); j++){
        Qpw(2, j) = gl(0, j);
        Qpw(0, j) = r * cos(gl(0, j) * 2 * M_PI * dir);
        Qpw(1, j) = r * sin(gl(0, j) * 2 * M_PI * dir); 
        Qpw(3, j) = gl(1, j);
    }

    // multiply weights with the normal of derivatives on the quad points
    for(int j = 0; j < Qpw.cols(); j++){
        double dx = - 2 * M_PI * dir * r * sin(Qpw(2, j) * 2 * M_PI * dir);
        double dy = 2 * M_PI * dir * r * cos(Qpw(2, j) * 2 * M_PI * dir);
        Qpw(4, j) = gl(1, j) * sqrt(pow(dx, 2) + pow(dy, 2));
    }

    return Qpw;
}

MatrixXd CellQuadPts(double r, double start, double end, double x_top, double y_top, int deg, int Ngl){
    assert(start < end);
    double dir = (r == 0.1) ? -1.0 : 1.0;
    MatrixXd GL_lmd = CrvQuadPts(r, start, end, Ngl + 1);
    MatrixXd GL_the = GaussQuad(0.0, 1.0, deg + 1);
    MatrixXd Qpw = MatrixXd::Zero(3, GL_lmd.cols() * GL_the.cols());
    int idx = 0;
    for (int j = 0; j < GL_the.cols(); j++){
        for (int i = 0; i < GL_lmd.cols(); i++){
            double x = GL_lmd(0, i);
            double y = GL_lmd(1, i);
            Qpw(0, idx) = (1.0 - GL_the(0, j)) * x + GL_the(0, j) * x_top;
            Qpw(1, idx) = (1.0 - GL_the(0, j)) * y + GL_the(0, j) * y_top;
            double dx = - 2 * M_PI * dir * r * sin(GL_lmd(2, i) * 2 * M_PI * dir);
            double dy = 2 * M_PI * dir * r * cos(GL_lmd(2, i) * 2 * M_PI * dir);
            Qpw(2, idx) = (1 - GL_the(0, j)) * (dx * (y_top - y) - dy * (x_top - x)) * GL_lmd(3, i) * GL_the(1, j);
            idx ++;
        }
    }
    return Qpw;
}

double QuadSum(VectorXd f, MatrixXd Qpw){
    int id = (Qpw.rows() == 5) ? 4 : 2;
    VectorXd w = Qpw.row(id);
    assert(f.size() == w.size());
    return f.dot(w);
}

/* compute the truncated factorial*/
double DivFactorial(int Ns, int Ne, int Ds, int De){
    double dF = 1.0;
    int NN = Ne - Ns + 1, ND = De - Ds + 1;
    for (int i = 1; i <= min(NN, ND); i ++){
        dF = dF * double(Ns + i - 1) / double(Ds + i -1);
    }
    if (NN > ND){
        for (int i = ND + 1; i<= NN; i ++){
            dF = dF * double(Ns + i - 1);
        }
    }
    else if (ND > NN){
        for (int i = NN + 1; i<= ND; i ++){
            dF = dF / double(Ds + i - 1);
        }
    }
    return dF;
}

/*===================================  basis =================================== */

/* The basis function over the reference triangular cell based on a uniform nodal distribution */
MatrixXd StandardBasis(int type, int deg){
    int dof; 
    if (type == 0){
        dof = deg + 1;
    }
    else if (type == 1){
        dof = (deg + 1) * (deg + 2) / 2;
    }
    MatrixXd Coef(dof, dof);
    Coef.setZero();

    switch (type)
    {
        case 0:
        {
            MatrixXd a(1,dof);
            int k;
            double b0, b1;
            for (int i = 0; i < dof; i++){
                Coef(i, 0) = 1.0;
                k = 0;
                for (int j = 0; j < dof; j++){
                    if (j != i){
                        a.block(0, 0, 1, k+1) = Coef.block(i, 0, 1, k+1);
                        b0 = double(2*(j+1)-dof-1) / double(2*(j-i));
                        b1 = - double(dof-1) / double(2*(j-i));

                        Coef(i, 0) = a(0, 0) * b0;
                        Coef(i, k+1) = a(0, k) * b1;
                        
                        for (int l = 1; l < k+1; l++){
                            Coef(i,l) = a(0, l) * b0 + a(0, l-1) * b1;
                        }
                        k = k+1;
                    }
                }
            }
        }
        break;
        case 1:
        {
            VectorXd a(dof), b(dof), temp(dof);
            int k;
            double d0, d1, d2;
            int pos, pos1, pos2;
            int cnt = 0;

            for(int j=0; j <= deg; j++){
                for(int i=0; i <= deg - j; i++){
                    k = deg - i - j;
                    a.setZero();
                    b.setZero();

                    if (i != 0){
                        a(1) = double(deg) / double(i);
                        if (i > 1){
                            for (int l=1; l<=i-1; l++){
                                temp = a;
                                d0 = double(l) / double(l-i);
                                d1 = - double(deg) / double(l-i);

                                a(0) = temp(0) * d0;
                                a((l+1)*(l+2)/2) = temp(l*(l+1)/2) * d1;

                                for(int m=1; m<=l ; m++){
                                    a(m*(m+1)/2) = temp(m*(m+1)/2) * d0 + temp((m-1)*m/2) * d1;
                                }
                            }
                        }
                    }
                    else{
                        a(0) = 1.0;
                    }

                    if (j != 0){
                        b(2) = double(deg) / double(j);
                        if (j > 1){
                            for (int t=1; t <= j-1; t++){
                                temp = b;
                                d0 = double(t) / double(t-j);
                                d1 = - double(deg) / double(t-j);

                                b(0) = temp(0) * d0;
                                b((t+2)*(t+3)/2-1) = temp((t+1)*(t+2)/2-1) * d1;

                                for(int s=1; s<=t ; s++){
                                    b((s+1)*(s+2)/2-1) = temp((s+1)*(s+2)/2-1) * d0 + temp(s*(s+1)/2-1) * d1;
                                }
                            }
                        }
                    }
                    else{
                        b(0) = 1.0;
                    }

                    for (int t = 0; t <= i; t++){
                        for (int s = 0; s <= j; s++){
                            pos = (t+s) * (t+s+1) / 2 + s + 1;
                            Coef(cnt, pos-1) = Coef(cnt, pos-1) + a(t*(t+1)/2) * b((s+1)*(s+2)/2-1);
                        }
                    }

                    if (k != 0){
                        for (int n=0; n<=k-1; n++){
                            temp = Coef.row(cnt);
                            d0 = double(n - deg) / double(n-k);
                            d1 = double(deg) / double(n-k);
                            d2 = double(deg) / double(n-k);

                            for(int c=0; c <= deg; c++){
                                for(int m=0; m<=c; m++){
                                    int l = c-m;
                                    pos = (l+m) * (l+m+1) / 2 + m + 1;
                                    pos1 = (l-1+m) * (l-1+m+1) / 2 + m + 1;
                                    pos2 = (l+m-1) * (l+m-1+1) / 2 + m - 1 + 1;

                                    Coef(cnt, pos-1) = temp(pos-1) * d0;
                                    if(l > 0){ Coef(cnt, pos-1) = Coef(cnt, pos-1) + temp(pos1-1) * d1; }
                                    if(m > 0){ Coef(cnt, pos-1) = Coef(cnt, pos-1) + temp(pos2-1) * d2; }
                                }
                            }
                        }
                    }
                    
                    cnt++;
                }
            }

        }
        break;
    }

    return Coef;
}

/* generate the grid nodes over the NURBS-enhanced triangles */

MatrixXd NodalDis(int deg, double r, double start, double end, double x_interior, double y_interior){
    assert(deg > 0 && start < end);
    double dir = (r == 0.1) ? -1.0 : 1.0;
    int Np = (deg + 1) * (deg + 2) / 2.0;
    // Generate uniform nodal distribution over a reference triangle (0,0)-(1,0)-(1,0)
    MatrixXd refNodes(2, Np);
    int idx = 0;
    for(int j = 0; j <= deg; j++){
        for(int i = 0; i <= deg - j; i++){
            refNodes(0, idx) = double(i) / double(deg);
            refNodes(1, idx) = double(j) / double(deg);
            idx ++;
        }
    }
    // map the nodes in reference triangle to the NURBS-enhanced triangle
    // (0, 0) to x3; (1, 0) to x1 = C(1); (0, 1) to x2
    MatrixXd Grid(2, Np);
    idx = 0;
    for(int j = 0; j <= deg; j++){
        for(int i = 0; i <= deg - j; i++){
            if(refNodes(0, idx) + refNodes(1, idx) == 0.0){
                Grid(0, idx) = x_interior;
                Grid(1, idx) = y_interior;
            }
            else{
                double lmd = refNodes(1, idx) / (refNodes(0, idx) + refNodes(1, idx));
                double the = 1 - ((refNodes(0, idx) + refNodes(1, idx)));
                double par = start + (end - start) * lmd;
                double x = r * cos(par * 2 * M_PI * dir);
                double y = r * sin(par * 2 * M_PI * dir);
                Grid(0, idx) = x_interior * the + (1 - the) * x; 
                Grid(1, idx) = y_interior * the + (1 - the) * y;
            } 
            idx ++;
        }
    }
    return Grid;
}

/* compute the coefficients of a set of Basis polynomias (complete Lagrange) over an arbitrary nodal distribution
   w.r.t. the order 1, x, y, x^2, xy, y^2, x^3, x^2y, ... ... */
MatrixXd cal_Basis(MatrixXd NodalDis, int deg){
    int dof = (deg + 1) * (deg + 2) / 2;
    if(dof != NodalDis.cols()){
        cout << "The number of nodes isn't consistent with the degree." << endl;
    }
    MatrixXd Coef = MatrixXd::Zero(dof, dof); // each row corresponds to the coefficients of a single LP
    MatrixXd M = MatrixXd::Zero(dof, dof);
    // VectorXd evl = VectorXd::Zero(dof);
    int idx = 0;
    for(int d = 0; d <= deg; d++){
        for(int dy = 0; dy <= d; dy++){
            int dx = d - dy;
            M.col(idx) = NodalDis.row(0).array().pow(dx) * NodalDis.row(1).array().pow(dy);
            idx ++;
        }
    }

    VectorXd b = VectorXd::Zero(dof);
    for(int i = 0; i < dof; i++){
        b.setZero();
        b(i) = 1.0;
        Coef.row(i) = M.lu().solve(b);
    }

    for(int a = 0; a < Coef.rows(); a++){
        for(int b = 0; b < Coef.cols(); b++){
            assert(!isnan(Coef(a, b)));
        }
    }

    return Coef;
}

/* evaluate a set of basis polynomials over a set of points */
MatrixXd evl_Basis(MatrixXd Coef, MatrixXd evlPts, int deg){
    int dof = (deg + 1) * (deg + 2) / 2;
    MatrixXd Monomials = MatrixXd::Zero(dof, evlPts.cols());
    int idx = 0;
    for(int d = 0; d <= deg; d++){
        for(int dy = 0; dy <= d; dy++){
            int dx = d - dy;
            Monomials.row(idx) = evlPts.row(0).array().pow(dx) * evlPts.row(1).array().pow(dy);
            idx ++;
        }
    }
    MatrixXd Values = Coef * Monomials;
    Values.transposeInPlace();
    return Values; // Np x DoF: column i stands for the values of i-th basis over Np points  
}

/* compute the coefficients of partial derivativers w.r.t. x of a set of basis polynomials (coefficients given)
   w.r.t. the order of 1, x, y, x^2, xy, y^2, ... ...*/
MatrixXd cal_dxBasis(MatrixXd Coef, int deg){
    int dof = deg * (deg + 1) / 2;
    MatrixXd dxLP = MatrixXd::Zero(Coef.rows(), dof);
    int idx1 = 0;
    int idx2 = 0;
    for(int d = 0; d <= deg; d++){
        for(int dy = 0; dy <= d; dy++){
            int dx = d - dy;
            if(dx == 0){
                idx1 ++;
                continue;
            }
            else{
                dxLP.col(idx2) = Coef.col(idx1).array() * dx;
                idx1 ++;
                idx2 ++;
            }
        }
    }
    return dxLP;
}

/* compute the coefficients of partial derivativers w.r.t. y of a set of basis polynomials (coefficients given)
   w.r.t. the order of 1, x, y, x^2, xy, y^2, ... ...*/
MatrixXd cal_dyBasis(MatrixXd Coef, int deg){
    int dof = deg * (deg + 1) / 2;
    MatrixXd dyLP = MatrixXd::Zero(Coef.rows(), dof);
    int idx1 = 0;
    int idx2 = 0;
    for(int d = 0; d <= deg; d++){
        for(int dy = 0; dy <= d; dy++){
            int dx = d - dy;
            if(dy == 0){
                idx1 ++;
                continue;
            }
            else{
                dyLP.col(idx2) = Coef.col(idx1).array() * dy;
                idx1 ++;
                idx2 ++;
            }
        }
    }
    return dyLP;
}

/* evaluate a set of partial der w.r.t. x (or y) of lagrange polynomials over a set of points */
MatrixXd evl_dBasis(MatrixXd dCoef, MatrixXd evlPts, int deg){
    int dof = deg * (deg + 1) / 2;
    assert(dof == dCoef.cols());
    MatrixXd Monomials = MatrixXd::Zero(dof, evlPts.cols());
    int idx = 0;
    for(int d = 0; d <= deg-1; d++){
        for(int dy = 0; dy <= d; dy++){
            int dx = d - dy;
            Monomials.row(idx) = evlPts.row(0).array().pow(dx) * evlPts.row(1).array().pow(dy);
            idx ++;
        }
    }
    MatrixXd Values =  dCoef * Monomials;
    Values.transposeInPlace();
    return Values; // Np x DoF: column i stands for the values of i-th basis der func over Np points
}