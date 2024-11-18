#include<iostream>
#include<fstream>
#include<math.h>
#include<string>
#include<vector>
#include<cassert>
#include<time.h>
#include <unordered_map>

#include "Eigen/Dense"
#include "Utility/Utility.h"
#include "VelocityMesh/VelocityMesh.h"
#include "SpatialMesh/SpatialMesh.h"
#include "Integration/Integration.h"
#include "MacroProperty/MacroProperty.h"
#include "DGSolver/DGSolver.h"
#include "Output/Output.h"

using namespace std;
using namespace Eigen;

int POLYDEG = 4;
int NPOLE = 1;             
int NAZIM = 40;
int NP_FC = 15;
int N_GL = 15;
int N_THE = 5;
int N_LMD = 8;

double CV = 1.0;
double VG = 1.0;

int TMAX = 8000000;
double TOL = 1.0e-7;

int main(){

    double TAU_N_INV = 100;
    VectorXd tau_r_inv(5);
    tau_r_inv << 1.0 / 0.001, 1.0 / 0.01, 1.0 / 0.1, 1.0 / 1.0, 1.0 / 10.0;

    MatrixXi BDINF(3, 11);
    BDINF << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // tag
                1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, // 0 - straight-sided bounary; 1 - curved boundary
                2, 3, 4, 3, 2, 3, 4, 3, 2, 2, 2; // 1 - thermalizing BC; 2 - diffusely reflecting BC; 3 - periodic BC; 4 - specularly reflecting BC
    VectorXd BD_TEMP(11);
    BD_TEMP << 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0;

    ofstream write_output("results/raw/ComputationRecord.dat");
    assert(write_output.is_open());
    write_output.setf(ios::fixed);
    write_output.precision(12);

    int MESHTYPE = 0;

    for(int m = 1; m <= 10; m++){

        string tag0 = to_string(m);

        for(int k = 1; k < 5; k++){

            double TAU_R_INV = tau_r_inv(k);
            double TAU_C_INV = TAU_R_INV + TAU_N_INV;

            cout << "====================== Mesh index: " << m << " =======================" << endl;
            cout << "====================== Kdnusen pair: " << k << " ===========================" << endl;
            
            cout << "=== === Pre-processing === ===" << endl;

            // input
            const char* STANDARDMESH = NULL;
            // output
            const char* OUTPUTPATH = NULL;
            
            string tag1 = to_string(k);
            string tag2 = to_string(MESHTYPE);
            string outpath = "results/raw/New_Porous_";
            string suffix = ".dat";
            string suffix2 = ".msh";
            string underline = "_";
            string path = "Mesh/Porous/porous_";
            string stdmsh = path + tag0 + suffix2;
            STANDARDMESH = stdmsh.c_str();
            string output = outpath + tag0 + underline + tag1 + underline + tag2 + suffix;
            OUTPUTPATH = output.c_str();

            /* Descretization of velocity space */
            VelocityMesh VMesh(NPOLE, NAZIM, VG, CV);
            MatrixXd dOmega = VMesh.GetdOmega();
            MatrixXd Cx = VMesh.GetCx();
            MatrixXd Cy = VMesh.GetCy();

            /* Generate spatial mesh */
            Mesh mesh(MESHTYPE, STANDARDMESH, BDINF, VMesh, POLYDEG, N_GL, N_THE, N_LMD);
            vector<Node> Nodes = mesh.GetNodes();
            vector<Face> Faces = mesh.GetFaces();
            vector<Cell> Cells = mesh.GetCells();
            vector<vector<vector<int>>> ComputeOrder = mesh.GetComputeOrder();
            int Nt = Cells.size();
            int Nf = Faces.size();
            int DoF3D = (POLYDEG + 1) * (POLYDEG + 2) / 2;

            /* =================================================================================================================== */

            /* Pre-compute the integrals */
            Integration IntPre(Nodes, Faces, Cells, dOmega, Cx, Cy, BDINF, POLYDEG, NP_FC, N_LMD, N_THE);
            Eigen::MatrixXd IntMat = IntPre.GetIntMat();
            vector<MatrixXd> MassMat = IntPre.GetMassMat();
            vector<MatrixXd> StfMatX = IntPre.GetStfMatX();
            vector<MatrixXd> StfMatY = IntPre.GetStfMatY();
            vector<vector<vector<vector<MatrixXd>>>> MassOnFace = IntPre.GetMassOnFace();
            vector<vector<vector<vector<MatrixXd>>>> MassOnFace_in = IntPre.GetMassOnFace_in();
            vector<vector<vector<vector<MatrixXd>>>> MassOnFace_out = IntPre.GetMassOnFace_out();
            vector<vector<vector<vector<MatrixXd>>>> FluxInt = IntPre.GetFluxInt();
            std::vector<std::vector<std::vector<Eigen::MatrixXd>>> IntOnFace = IntPre.GetIntOnFace();

            /* Initialize the macro property */
            MacroProperty MP(Nt, DoF3D);
            MatrixXd Ts, Qxs, Qys;
            unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> FW;
            vector<vector<MatrixXd>> EDF = vector<vector<MatrixXd>>(NPOLE, vector<MatrixXd>(NAZIM, MatrixXd::Zero(Nt, DoF3D)));

            /* Assemble the coefficient matrix */
            vector<vector<vector<MatrixXd>>> Asol = CoefficientMat(MassMat, StfMatX, StfMatY, MassOnFace_out, Cx, Cy, \
                                                                ComputeOrder, NPOLE, NAZIM, Nt, TAU_C_INV);

            cout << "=== === start computing === ===" << endl;
            int step = 1;
            double res = 10000;
            clock_t start, end;
            start = clock();
            while(step < TMAX && res > TOL){
                Ts = MP.GetTs();
                Qxs = MP.GetQxs();
                Qys = MP.GetQys();
                FW = Cal_Flux_Wall(EDF, Faces, Cells, Cx, Cy, dOmega, BDINF, MassOnFace, NPOLE, NAZIM, Nf, DoF3D, POLYDEG);

                vector<vector<MatrixXd>> EDFn = EDF;
                EDF = DG_Solver_EDF(ComputeOrder, Faces, Cells, MassMat, MassOnFace_in, FluxInt, IntOnFace, FW, Asol, EDF, EDFn, \
                                    Ts, Qxs, Qys, Cx, Cy, BD_TEMP, BDINF, \
                                    POLYDEG, DoF3D, Nt, NPOLE, NAZIM, CV, VG, TAU_R_INV, TAU_N_INV);
                
                MP.Cal_MacroProperty(EDF, Faces, Cells, Cx, Cy, dOmega, IntMat, Nt, DoF3D, CV);
                res = MP.Cal_ResidualT();
                cout << "Step " << step << " : res = "<< res << ";" << endl;
                step++;
            }

            end = clock();

            double times;
            times = (double)(end - start)/CLOCKS_PER_SEC;
            cout << "Time: " << times << endl;

            Output(MESHTYPE, EDF, Nodes, Faces, Cells, dOmega, Cx, Cy, 501, CV, POLYDEG, OUTPUTPATH);

            write_output << "------------------------------------------------------------------" << endl;
            write_output << "Mesh idex: " << m << "; MeshType: " << MESHTYPE << endl;
            write_output << "Kndusen idx: " << k << ": " << "TAU_N_INV = " << TAU_N_INV << ", TAU_R_INV = " << TAU_R_INV << endl;
            write_output << "Number of Elements = " << Nt << endl;
            write_output << "# Iteration Steps = " << step << endl;
            write_output << "CPU time = " << times << endl;
            write_output << "------------------------------------------------------------------" << endl << endl;
        }
    }


    return 0;
}