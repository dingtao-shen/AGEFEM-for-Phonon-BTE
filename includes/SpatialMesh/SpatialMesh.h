#ifndef MESH
#define MESH

#include "data_struct.h"
#include "VelocityMesh/VelocityMesh.h"
#include "Eigen/Dense"

class Mesh
{
private:
    std::vector<Node> Nodes;
    std::vector<Face> Faces;
    std::vector<Cell> Cells;
    std::vector<std::vector<std::vector<int>>> ComputeOrder;
    double minCellSize;
public:
    Mesh(int type, const std::string &standardmesh, Eigen::MatrixXi boundaryinf, VelocityMesh VelMesh, \
         int deg, int Ngl, int Nthe, int Nlmd);
    ~Mesh();
    std::vector<Node> GetNodes();
    std::vector<Face> GetFaces();
    std::vector<Cell> GetCells();
    std::vector<std::vector<std::vector<int>>> GetComputeOrder();
    double GetMinCellSize();
};

#endif