#ifndef OUTPUT
#define OUTPUT

#include <vector>
#include "Eigen/Dense"
#include "SpatialMesh/data_struct.h"

void Output(int meshtype, std::vector<std::vector<Eigen::MatrixXd>> EDF, std::vector<Node> Nodes, std::vector<Face> Faces, std::vector<Cell> Cells, \
            Eigen::MatrixXd dOmega, Eigen::MatrixXd Cx, Eigen::MatrixXd Cy, int N, double Cv, int deg, const std::string &Outputpath);

#endif