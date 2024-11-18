#include<iostream>
#include<math.h>
#include<cassert>
#include<fstream>
#include<string>
#include <unordered_map>

#include "Utility/Utility.h"
#include "SpatialMesh/data_struct.h"
#include "SpatialMesh/SpatialMesh.h"
#include "VelocityMesh/VelocityMesh.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

Mesh::Mesh(int type, const std::string &standardmesh, Eigen::MatrixXi boundaryinf, VelocityMesh VelMesh, int deg, int Ngl, int Nthe, int Nlmd){
    /* Read in standard straight-sided mesh */
    ifstream readmesh(standardmesh);
    // if (!readmesh.is_open()) {
    //     perror("Failed to open file");
    // }
    assert(readmesh.is_open());
    std::cout << ">>> Read Spatial Mesh:" << endl;

    // read nodes
    string line;
    while (getline(readmesh, line)) {
        if (line.find("$Nodes") != string::npos) {
            break;
        }
    }
    std::cout << "   >>> Read Nodes... " << endl;
    int Nn;
    int dummy;
    readmesh >> Nn;
    Nodes.resize(Nn);
    for(int i = 0; i < Nn; i++){
        readmesh >> dummy >>  Nodes[i].x >> Nodes[i].y >> dummy;
        Nodes[i].index = i;
    }

    // read faces and cells
    while (getline(readmesh, line)) {
        if (line.find("$Elements") != string::npos) {
            break;
        }
    }
    std::cout << "   >>> Read Elements... " << endl;
    int Ne = 0, Nt = 0, Nf = 0, n1, n2;
    bool inque;
    readmesh >> Ne;
    MatrixXi Elements = MatrixXi::Zero(Ne, 5);
    MatrixXi Face_tmp = MatrixXi::Zero(Ne * 3, 4);
    VectorXi nd = VectorXi::Zero(4);
    for(int i = 0; i < Ne; i++){
        readmesh >> dummy >> Elements(i, 0) >> dummy >> Elements(i, 1);
        // store the face element (they're not the total faces)
        if (Elements(i, 0) == 1) {
            readmesh >> dummy >> Elements(i, 2) >> Elements(i, 3);
            // the input index is 1-based, convert it to 0-based
            Elements(i, 2)--;
            Elements(i, 3)--;
            Nf++;
            Face_tmp.block(Nf-1, 1, 1, 2) = Elements.block(i, 2, 1, 2);
            for(int j = 0; j < boundaryinf.cols(); j++){
                if (Elements(i, 1) == boundaryinf(0, j)){
                    Face_tmp(Nf-1, 0) = boundaryinf(0, j);
                    continue;
                }
            }
        }
        // store the triangle
        else if(Elements(i, 0) == 2){
            readmesh >> dummy >> Elements(i, 2) >> Elements(i, 3) >> Elements(i, 4);
            // the input index is 1-based, modify it to 0-based
            Elements(i, 2)--;
            Elements(i, 3)--;
            Elements(i, 4)--;
            Nt++;
            for(int k = 0; k < 3; k++){nd(k) = Elements(i, k+2);}
            nd(3) = nd(0);
            // extract (interior) faces from the cells
            for(int j = 0; j < 3; j++){
                inque = false;
                // check if the edge has been stored in Face_temp
                for(int k = 0; k < Nf; k++){
                    n1 = Face_tmp(k, 1);
                    n2 = Face_tmp(k, 2);
                    if((n1 == nd(j)) && (n2 == nd(j+1))){
                        inque = true;
                        continue;
                    }
                    if((n1 == nd(j+1)) && (n2 == nd(j))){
                        inque = true;
                        continue;
                    }
                }
                if(!inque){
                    Nf++;
                    Face_tmp(Nf-1, 1) = nd(j);
                    Face_tmp(Nf-1, 2) = nd(j+1);
                }
            }
        }
    }

    // read periodic pairs
    int Npair = 0;
    unordered_map<int, int> NodePairs24, NodePairs73, NodePairs86;
    unordered_map<int, int> BDPairs;
    int nM, nS, bM, bS;
    while (getline(readmesh, line)) {
        if (line.find("$Periodic") != string::npos) {
            break;
        }
    }
    std::cout << "   >>> Read Periodic Pairs... " << endl;
    readmesh >> dummy;

    readmesh >> dummy >> bS >> bM;
    BDPairs[bS] = bM;
    readmesh >> line;
    for(int d = 0; d < 16; d++){readmesh >> dummy;}
    readmesh >> Npair;
    for(int p = 0; p < Npair; p++){
        readmesh >> nS >> nM;
        NodePairs24[nS - 1] = nM - 1;
    }

    readmesh >> dummy >> bS >> bM;
    BDPairs[bS] = bM;
    readmesh >> line;
    for(int d = 0; d < 16; d++){readmesh >> dummy;}
    readmesh >> Npair;
    for(int p = 0; p < Npair; p++){
        readmesh >> nS >> nM;
        NodePairs86[nS - 1] = nM - 1;
    }

    // set faces and cells
    std::cout << "   >>> Construct the Faces and Cells information... " << endl;
    Faces.resize(Nf);
    Cells.resize(Nt);
    double hmin_tmp;
    minCellSize = 1.0;
    VectorXd nx = VectorXd::Zero(4);
    VectorXd ny = VectorXd::Zero(4);
    int idx = 0;
    for(int i = 0; i < Ne; i++){
        if(Elements(i, 0) == 2){
            for(int k = 0; k < 3; k++){
                Cells[idx].vertexes.push_back(Elements(i, 2+k));
            }
            for(int k = 0; k < 3; k++){
                nx(k) = Nodes[Cells[idx].vertexes[k]].x;
                ny(k) = Nodes[Cells[idx].vertexes[k]].y;
            }
            nx(3) = nx(0);
            ny(3) = ny(0);
            // the area of cell
            Cells[idx].area = 0.5 * (nx(2)*ny(0) - nx(1)*ny(0) + nx(0)*ny(1) - nx(2)*ny(1) - nx(0)*ny(2) + nx(1)*ny(2));
            // outward noraml vector over straight-sided face
            Cells[idx].str_outward_norm_vec.resize(2, 3);
            Cells[idx].str_outward_norm_vec.setZero();
            for(int k = 0; k < 3; k++){
                Cells[idx].str_outward_norm_vec(0, k) = (ny(k + 1) - ny(k)) / sqrt(pow(nx(k) - nx(k + 1), 2) + pow(ny(k) - ny(k + 1), 2));
                Cells[idx].str_outward_norm_vec(1, k) = (nx(k) - nx(k + 1)) / sqrt(pow(nx(k) - nx(k + 1), 2) + pow(ny(k) - ny(k + 1), 2));
                hmin_tmp = 2.0 * Cells[idx].area / sqrt( pow(nx(k) - nx(k + 1), 2) + pow(ny(k) - ny(k + 1), 2));
                minCellSize = (minCellSize > hmin_tmp) ? hmin_tmp : minCellSize;
            }
            idx ++;
        }
    }
    assert(idx == Nt); // check all the cells are set
    
    VectorXd px(2), py(2);
    if(type == 0){
        for(int i = 0; i < Nf; i++){
            Faces[i].index = i;
            Faces[i].neighbor_cell.resize(2, -1);
            assert(Faces[i].neighbor_cell[0] == -1 && Faces[i].neighbor_cell[1] == -1);
            Faces[i].local_indices.resize(2, -1);
            assert(Faces[i].local_indices[0] == -1 && Faces[i].local_indices[1] == -1);

            Faces[i].type = 0;
            Faces[i].boundaryflag = Face_tmp(i, 0);
            Faces[i].vertexes.push_back(Face_tmp(i, 1));
            Faces[i].vertexes.push_back(Face_tmp(i, 2));
            for(int k = 0; k < 2; k++){
                px(k) = Nodes[Faces[i].vertexes[k]].x;
                py(k) = Nodes[Faces[i].vertexes[k]].y;
            }
            Faces[i].length = sqrt(pow((px(0) - px(1)), 2) + pow((py(0) - py(1)), 2));
        }
    }
    else if(type == 1){
        for(int i = 0; i < Nf; i++){
            Faces[i].index = i;
            Faces[i].neighbor_cell.resize(2, -1);
            assert(Faces[i].neighbor_cell[0] == -1 && Faces[i].neighbor_cell[1] == -1);
            Faces[i].local_indices.resize(2, -1);
            assert(Faces[i].local_indices[0] == -1 && Faces[i].local_indices[1] == -1);

            if(Face_tmp(i, 0) > 0 && boundaryinf(1, Face_tmp(i, 0) - 1) == 1){ // curved boundary
                Faces[i].type = 1;
                Faces[i].boundaryflag = Face_tmp(i, 0);
                Faces[i].vertexes.push_back(Face_tmp(i, 1));
                Faces[i].vertexes.push_back(Face_tmp(i, 2));
            }
            else{ // straight-sided boundary or inner faces
                Faces[i].type = 0;
                Faces[i].boundaryflag = Face_tmp(i, 0);
                Faces[i].vertexes.push_back(Face_tmp(i, 1));
                Faces[i].vertexes.push_back(Face_tmp(i, 2));
                for(int k = 0; k < 2; k++){
                    px(k) = Nodes[Faces[i].vertexes[k]].x;
                    py(k) = Nodes[Faces[i].vertexes[k]].y;
                }
                Faces[i].length = sqrt(pow((px(0) - px(1)), 2) + pow((py(0) - py(1)), 2));
            }
        }
    }

    // connect the faces and cells
    for(int i = 0; i < Nt; i++){
        Cells[i].faces.resize(3, -1);
        assert(Cells[i].faces[0] == -1 && Cells[i].faces[1] == -1 && Cells[i].faces[2] == -1);
        for(int k = 0; k < 3; k++){
            nd(k) = Cells[i].vertexes[k];
        }
        nd(3) = nd(0);
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < Nf; k++){
                n1 = Faces[k].vertexes[0];
                n2 = Faces[k].vertexes[1];
                if(nd(j) == n1 && nd(j + 1) == n2){
                    Cells[i].faces[j] = k;
                    Faces[k].neighbor_cell[0] = i;
                    Faces[k].local_indices[0] = j;
                    continue;
                }
                if(nd(j) == n2 && nd(j + 1) == n1){
                    Cells[i].faces[j] = k;
                    Faces[k].neighbor_cell[1] = i;
                    Faces[k].local_indices[1] = j;
                    continue;
                }
            }
        }
    }

    /* Enhance the faces along the curved boundary by exact circle information */
    double rs = 0.1, rl = 0.25;
    Circle H1(0.5, -1.0, rl, 1);
    Circle H5(0.5, 1.0, rl, 1);
    Circle H9(-0.5, -0.5, rl, 1);
    Circle H10(-0.5, 0.5, rs, 1);
    Circle H11(0.5, 0.0, rs, 1);
    unordered_map<int, Circle> HoleSet;
    HoleSet[1] = H1;
    HoleSet[5] = H5;
    HoleSet[9] = H9;
    HoleSet[10] = H10;
    HoleSet[11] = H11;

    for(int i = 0; i < Nt; i++){
        for(int j = 0; j < 3; j++){
            int FID = Cells[i].faces[j];
            double s, e, x_top, y_top;
            if(Faces[FID].type == 1){
                Circle H = HoleSet[Faces[FID].boundaryflag];
                double x1 = Nodes[Faces[FID].vertexes[0]].x;
                double y1 = Nodes[Faces[FID].vertexes[0]].y;
                double x2 = Nodes[Faces[FID].vertexes[1]].x;
                double y2 = Nodes[Faces[FID].vertexes[1]].y;
                double p1 = H.CalPara(x1, y1);
                double p2 = H.CalPara(x2, y2);
                if((p1 < p2 && p1 > 0) or (p1 < p2 && p1 == 0 && p2 < 0.5)){
                    Faces[FID].nurbs_paras.push_back(p1);
                    Faces[FID].nurbs_paras.push_back(p2);
                    s = p1;
                    e = p2;
                    Faces[FID].QuadPts = H.ArcQuadPts(s, e, Ngl);
                }
                else if((p1 > p2 && p2 > 0) or (p1 > p2 && p2 == 0 && p1 < 0.5)){
                    swap(Faces[FID].vertexes[0], Faces[FID].vertexes[1]);
                    swap(Faces[FID].local_indices[0], Faces[FID].local_indices[1]);
                    swap(Faces[FID].neighbor_cell[0], Faces[FID].neighbor_cell[1]);
                    Faces[FID].nurbs_paras.push_back(p2);
                    Faces[FID].nurbs_paras.push_back(p1); 
                    s = p2;
                    e = p1;
                    Faces[FID].QuadPts = H.ArcQuadPts(s, e, Ngl);
                }
                else if((p1 > p2 && p2 == 0 && p1 > 0.5)){
                    swap(Faces[FID].vertexes[0], Faces[FID].vertexes[1]);
                    swap(Faces[FID].local_indices[0], Faces[FID].local_indices[1]);
                    swap(Faces[FID].neighbor_cell[0], Faces[FID].neighbor_cell[1]);
                    p2 = 1.0;
                    Faces[FID].nurbs_paras.push_back(p1);
                    Faces[FID].nurbs_paras.push_back(p2);
                    s = p1;
                    e = p2;
                    Faces[FID].QuadPts = H.ArcQuadPts(s, e, Ngl);
                }
                else if(p1 < p2 && p1 == 0 && p2 > 0.5){
                    swap(Faces[FID].vertexes[0], Faces[FID].vertexes[1]);
                    swap(Faces[FID].local_indices[0], Faces[FID].local_indices[1]);
                    swap(Faces[FID].neighbor_cell[0], Faces[FID].neighbor_cell[1]);
                    p1 = 1.0;
                    Faces[FID].nurbs_paras.push_back(p2);
                    Faces[FID].nurbs_paras.push_back(p1);
                    s = p2;
                    e = p1;
                    Faces[FID].QuadPts = H.ArcQuadPts(s, e, Ngl);
                }

                // cout << Faces[FID].boundaryflag << "; " << FID << "; " << Faces[FID].nurbs_paras[0] << "; " << Faces[FID].nurbs_paras[1] << endl;

                // compute the length of the NURBS enhanced face
                VectorXd f_ones = VectorXd::Ones(Faces[FID].QuadPts.cols());
                Faces[FID].length = QuadSum(f_ones, Faces[FID].QuadPts);

                Cells[i].type = 1;
                // compute the quadrature points of the NURBS-enhanced cell
                for(int k = 0; k < 3; k++){
                    if(Cells[i].vertexes[k] != Faces[FID].vertexes[0] && Cells[i].vertexes[k] != Faces[FID].vertexes[1]){
                        x_top = Nodes[Cells[i].vertexes[k]].x;
                        y_top = Nodes[Cells[i].vertexes[k]].y;
                        break;
                    }
                }
                
                Cells[i].IntpNds = H.NbrCell_NodalDis(deg, s, e, x_top, y_top);
                Cells[i].Basis = cal_Basis(Cells[i].IntpNds, deg);

                Cells[i].QuadPts = H.NbrCell_QuadPts(s, e, x_top, y_top, deg, Ngl);
                //compute the area of NURBS enhanced cell
                VectorXd F_ones = VectorXd::Ones(Cells[i].QuadPts.cols());
                Cells[i].area = QuadSum(F_ones, Cells[i].QuadPts);

                // set the outward normal vector over the nurbs curve
                Cells[i].crv_outward_norm_vec.resize(2, Faces[FID].QuadPts.cols());
                Cells[i].crv_outward_norm_vec.setZero();
                for(int k = 0; k < Faces[FID].QuadPts.cols(); k++){
                    Cells[i].crv_outward_norm_vec(0, k) = H.CalNVec(Faces[FID].QuadPts(2, k))[0];
                    Cells[i].crv_outward_norm_vec(1, k) = H.CalNVec(Faces[FID].QuadPts(2, k))[1];
                }
                break;
            }
        }
    }

    /* construct the computation order of the mesh, based on the velocity mesh information */

    std::cout << ">>> The spatial mesh is ready!" << endl;

    std::cout << ">>> Set the computation order..." << endl;
    MatrixXd dOmega = VelMesh.GetdOmega();
    MatrixXd Cx = VelMesh.GetCx();
    MatrixXd Cy = VelMesh.GetCy();
    int Np = dOmega.rows();
    int Na = dOmega.cols();
    ComputeOrder.resize(Np, vector<vector<int>>(Na, vector<int>(Nt, -1)));
    
    int FCID, CellID1, CellID2;
    for(int i = 0; i < Nt; i++){
        Cells[i].neighbour_cells.resize(3, -1);
        assert(Cells[i].neighbour_cells[0] == -1 && Cells[i].neighbour_cells[1] == -1 && Cells[i].neighbour_cells[2] == -1);
        for(int k = 0; k < 3; k++){
            FCID = Cells[i].faces[k];
            CellID1 = Faces[FCID].neighbor_cell[0];
            CellID2 = Faces[FCID].neighbor_cell[1];
            if(CellID1 == i){Cells[i].neighbour_cells[k] = CellID2; }
            if(CellID2 == i){Cells[i].neighbour_cells[k] = CellID1; }
        }
    }

    MatrixXi TriIOFlag(Nt, 3);
    int cnt;
    bool outgoing;
    double vx, vy, flux;
    for(int j1 = 0; j1 < Np; j1++){
        for(int j2 = 0; j2 < Na; j2++){
            TriIOFlag.setZero();
            for(int i = 0; i < Nt; i++){
                for(int k = 0; k < 3; k++){
                    if(Cells[i].neighbour_cells[k] != -1){
                        vx = Cells[i].str_outward_norm_vec(0, k);
                        vy = Cells[i].str_outward_norm_vec(1, k);
                        flux = Cx(j1, j2) * vx + Cy(j1, j2) * vy;
                        if(flux > 0.0){TriIOFlag(i, k) = 1;}
                        if(flux < 0.0){TriIOFlag(i, k) = -1;}
                    }
                }
            }
            vector<bool> InArray(Nt, true);
            cnt = 0;
            do{
                for(int i = 0; i < Nt; i++){
                    if(InArray[i]){
                        outgoing = true;
                        for(int k = 0; k < 3; k++){
                            if(TriIOFlag(i, k) == -1){
                                outgoing = false;
                                break;
                            }
                        }
                        if(outgoing){
                            cnt++;
                            ComputeOrder[j1][j2][cnt - 1] = i;
                            InArray[i] = false;
                            for(int k = 0; k < 3; k++){
                                CellID1 = Cells[i].neighbour_cells[k];
                                if(CellID1 != -1){
                                    for(int idx = 0; idx < 3; idx ++){
                                        if(Cells[CellID1].neighbour_cells[idx] == i){
                                            TriIOFlag(CellID1, idx) = 0;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }while(cnt != Nt);
        }
    }
    std::cout << ">>> The computation order has been set." << endl;

    // modify the connect relationship by the periodic pairs
    for(int i = 0; i < Nf; i++){
        if(Faces[i].boundaryflag == 2 || Faces[i].boundaryflag == 8){

            unordered_map<int, int> NodePairs;
            if(Faces[i].boundaryflag == 2){
                NodePairs = NodePairs24;
            }
            else if(Faces[i].boundaryflag == 8){
                NodePairs = NodePairs86;
            }
            
            int bm = BDPairs[Faces[i].boundaryflag];
            int nm1 = NodePairs[Faces[i].vertexes[0]];
            int nm2 = NodePairs[Faces[i].vertexes[1]];
            int v1 = (Faces[i].neighbor_cell[1] == -1) ? 1 : 0;
            for(int j = 0; j < Nf; j++){
                if(Faces[j].boundaryflag == bm \
                    && ((Faces[j].vertexes[0] == nm1 && Faces[j].vertexes[1] == nm2) \
                        || (Faces[j].vertexes[0] == nm2 && Faces[j].vertexes[1] == nm1))){
                        Faces[i].pair = Faces[j].index;
                        Faces[j].pair = Faces[i].index;
                        int v2 = (Faces[j].neighbor_cell[1] == -1) ? 1 : 0;
                        Faces[i].neighbor_cell[v1] = Faces[j].neighbor_cell[1 - v2];
                        Faces[j].neighbor_cell[v2] = Faces[i].neighbor_cell[1 - v1];
                        break;
                }
                else{
                    continue;
                }
            }
        }
        else{
            continue;
        }
    }

}

Mesh::~Mesh(){}

std::vector<Node> Mesh::GetNodes(){ return Nodes; }
std::vector<Face> Mesh::GetFaces(){ return Faces; }
std::vector<Cell> Mesh::GetCells(){ return Cells; }
std::vector<std::vector<std::vector<int>>> Mesh::GetComputeOrder(){ return ComputeOrder; };
double Mesh::GetMinCellSize(){ return minCellSize; }