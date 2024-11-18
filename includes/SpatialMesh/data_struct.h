#ifndef DATASTRUCT
#define DATASTRUCT

#include <cmath>
#include <vector>

#include "Eigen/Dense"
#include "Utility/Utility.h"

class Node {
public:
    int index;
    double x;
    double y;
};

class Face {
public:
    int index;
    int type; // 0 - straight-sided face; 1 - curved face
    double length;
    int boundaryflag; // 0 - interior face; >0 - the index of the boundary condition
    int pair; // index of paired face (this is a face on the periodic boundary);
    std::vector<int> vertexes; // indices of 2 vertexes a, b
    std::vector<int> neighbor_cell; // indices of 2 neighbor cells (+) (-), unordered
    std::vector<int> local_indices; // local indices of this faces in neighbor cells, 
    /*       b
             |
      (+)    |    (-)
             |
             a
    curl right hand fingers from a to b, right thumb points to a */
    std::vector<double> nurbs_paras; // start and end parameters of (the same one) NURBS curve
    Eigen::MatrixXd nurbs_points; // the coordinates of the points over the NURBS curve, if this faces is NURBS curve; 2 by x
    // nurbs_points may be not useful
    Eigen::MatrixXd QuadPts; // Gauss Quadrature Points along the face; 2 by x
};

class Cell {
public:
    int index;
    int type; // 0 - straight-sided cell; 1 - nurbs-enhanced cell
    double area;
    std::vector<int> vertexes; // indices of 3 vertexes a-b-c, in counter-clockwise order
    std::vector<int> faces; // indices of 3 faces of this cell: face1-face2-face3
    /*     c
           | \
    face3  |  \ face 2
           |   \ 
           a----b
           face1
    */
    std::vector<int> neighbour_cells; // indices of 3 neighbour cells. order w.r.t. faces;
    Eigen::MatrixXd str_outward_norm_vec; // each column - the outward normal vector of each straight-sided face; 2 by 3
    Eigen::MatrixXd crv_outward_norm_vec; // each column - the outward normal vector of each NURBS face; 2 by x (x is the same as cols of quadrature points)
    // remark: here we only consider the case each cell has at most 1 NURBS edge (type == 1)
    Eigen::MatrixXd IntpNds; // Intepolation nodes for NURBS-enhanced triangular cell;
    Eigen::MatrixXd Basis; // Coefficients of Lagrange polynomials based on the Intepolation nodes;
    Eigen::MatrixXd QuadPts; // Gauss quadrature points over the cell; 4 by x - only for nurbs-enhanced cell
};

class Circle
{
private:
    std::vector<double> origin;
    double radius;
    int direction; // 1 : clockwise or  -1 : counter-clockwise
public:

    Circle(){
        origin.push_back(0.0);
        origin.push_back(0.0);
        radius = 1;
        direction = 1;
    }

    Circle(double x0, double y0, double r, int dir){
        origin.push_back(x0);
        origin.push_back(y0);
        radius = r;
        direction = dir;
    };

    ~Circle(){};

    double CalPara(double x, double y){
        double p;
        double theta = asin((y - origin[1]) / radius); // -pi/2 to pi/2
        if(x >= origin[0] && y > origin[1]){
            p = (2 * M_PI - theta) / (2 * M_PI);
        }
        else if(x > origin[0] && y <= origin[1]){
            p = - theta / (2 * M_PI);
        }
        else if(x <= origin[0] && y < origin[1]){
            p = (M_PI + theta) / (2 * M_PI); 
        }
        else if(x < origin[0] && y >= origin[1]){
            p = (M_PI + theta) / (2 * M_PI);
        }
        assert(p >= 0.0 && p <= 1.0);
        return (direction == 1) ? p : 1.0-p;
    };

    std::vector<double> CalCrd(double p){
        std::vector<double> nd(2, 0.0);
        double theta = (direction == 1) ? (- p * 2.0 * M_PI) : (p * 2.0 * M_PI);
        nd[0] = origin[0] + radius * cos(theta);
        nd[1] = origin[1] + radius * sin(theta);
        return nd;
    };

    std::vector<double> CalDer(double p){ // derivative w.r.t. parameter p
        // pointing to the origin
        std::vector<double> der(2, 0.0);
        double theta = (direction == 1) ? (- p * 2.0 * M_PI) : (p * 2.0 * M_PI);
        double w = (direction == 1) ? (- 2.0 * M_PI * radius) : (2.0 * M_PI * radius);
        der[0] = - w * sin(theta);
        der[1] = w * cos(theta);
        return der;
    };

    std::vector<double> CalNVec(double p){
        // pointing to the origin
        std::vector<double> nv(2, 0.0);
        double theta = (direction == 1) ? (M_PI - p * 2.0 * M_PI) : (M_PI + p * 2.0 * M_PI);
        nv[0] = cos(theta);
        nv[1] = sin(theta);
        return nv;
    };

    Eigen::MatrixXd ArcQuadPts(double start, double end, int ngl){
        assert(start < end);
        Eigen::MatrixXd gl = GaussQuad(start, end, ngl);

        Eigen::MatrixXd Qpw = Eigen::MatrixXd::Zero(5, gl.cols());
        for (int j = 0; j < gl.cols(); j++){
            Qpw(2, j) = gl(0, j);
            Qpw(0, j) = CalCrd(gl(0, j))[0];
            Qpw(1, j) = CalCrd(gl(0, j))[1]; 
            Qpw(3, j) = gl(1, j);
        }

        // multiply weights with the normal of derivatives on the quad points
        for(int j = 0; j < Qpw.cols(); j++){
            double dx = CalDer(Qpw(2, j))[0];
            double dy = CalDer(Qpw(2, j))[1];
            Qpw(4, j) = gl(1, j) * sqrt(pow(dx, 2) + pow(dy, 2));
        }

        return Qpw;
    };

    // compute the nodal distribution of the cell that shares an arc with this cirle
    Eigen::MatrixXd NbrCell_NodalDis(int deg, double start, double end, double x_interior, double y_interior){
        assert(deg > 0 && start < end);
        int Np = (deg + 1) * (deg + 2) / 2.0;
        // Generate uniform nodal distribution over a reference triangle (0,0)-(1,0)-(1,0)
        Eigen::MatrixXd refNodes(2, Np);
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
        Eigen::MatrixXd Grid(2, Np);
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
                    double x = origin[0] + radius * cos(- par * 2 * M_PI * direction);
                    double y = origin[1] + radius * sin(- par * 2 * M_PI * direction);
                    Grid(0, idx) = x_interior * the + (1 - the) * x; 
                    Grid(1, idx) = y_interior * the + (1 - the) * y;
                } 
                idx ++;
            }
        }
        return Grid;
    };

    Eigen::MatrixXd NbrCell_QuadPts(double start, double end, double x_interior, double y_interior, int deg, int ngl){
        assert(start < end);
        Eigen::MatrixXd GL_lmd = ArcQuadPts(start, end, ngl + 1);
        Eigen::MatrixXd GL_the = GaussQuad(0.0, 1.0, deg + 1);
        Eigen::MatrixXd Qpw = Eigen::MatrixXd::Zero(3, GL_lmd.cols() * GL_the.cols());
        int idx = 0;
        for (int j = 0; j < GL_the.cols(); j++){
            for (int i = 0; i < GL_lmd.cols(); i++){
                double x = GL_lmd(0, i);
                double y = GL_lmd(1, i);
                Qpw(0, idx) = (1.0 - GL_the(0, j)) * x + GL_the(0, j) * x_interior;
                Qpw(1, idx) = (1.0 - GL_the(0, j)) * y + GL_the(0, j) * y_interior;
                double dx = CalDer(GL_lmd(2, i))[0];
                double dy = CalDer(GL_lmd(2, i))[1];
                Qpw(2, idx) = (1 - GL_the(0, j)) * (dx * (y_interior - y) - dy * (x_interior - x)) * GL_lmd(3, i) * GL_the(1, j);
                idx ++;
            }
        }
        return Qpw;
    };
};


#endif