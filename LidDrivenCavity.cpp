#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

/**
 * @brief Macro to map matrix entry i,j onto it's corresponding location in memory, assuming column-wise matrix storage
 * @param I     matrix index i denoting the ith row
 * @param J     matrix index j denoting the jth columns
 */
#define IDX(I,J) ((J)*Nx + (I))                     //define a new operation to improve computation?

#include "LidDrivenCavity.h"
#include "SolverCG.h"

LidDrivenCavity::LidDrivenCavity(MPI_Comm &rowGrid, MPI_Comm &colGrid, int coords0, int coords1)
{
    comm_row_grid = rowGrid;
    comm_col_grid = colGrid;
    MPIcoords[0] = coords0;                 //note that coordinates describe the grid index in matrix notation
    MPIcoords[1] = coords1;
    MPI_Comm_size(comm_row_grid,&size);     //get size of communicator
    
    //reduce global values onto all grid only, for correct calculation of dx and dy and printing of Lx,Ly,Nx,Ny etc.
    MPI_Allreduce(&Nx,&globalNx,1,MPI_INT,MPI_SUM,comm_row_grid);
    MPI_Allreduce(&Ny,&globalNy,1,MPI_INT,MPI_SUM,comm_col_grid);   //-> note for future, is there any point in splitting Lx Ly up in main, when I'm gonna need global values anyway?

    //compute ranks along the row communciator and along teh column communicator
    MPI_Comm_rank(comm_row_grid, &rowRank); 
    MPI_Comm_rank(comm_col_grid, &colRank);

    //compute ranks for adjacent grids for data transfer, if at boundary, returns -2 (MPI_PROC_NULL)
    MPI_Cart_shift(comm_col_grid,0,1,&topRank,&bottomRank);
    MPI_Cart_shift(comm_row_grid,0,1,&leftRank,&rightRank);
    
    //cout << "Coord (" << coords0 << "," << coords1 << ") has row rank " << rowRank << " and left is " << leftRank << " and right is "<< right << endl;
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();                                                      //deallocate memory
}

    //getting functions for testing purposes
double LidDrivenCavity::GetDt(){
    return dt;
} 

double LidDrivenCavity::GetT() {
    return T;
}

double LidDrivenCavity::GetDx() {
    return dx;
}   

double LidDrivenCavity::GetDy() {
    return dy;
}   
    
int LidDrivenCavity::GetNx() {
    return globalNx;
}

int LidDrivenCavity::GetNy() {
    return globalNy;
}

int LidDrivenCavity::GetNpts() {
    return globalNx*globalNy;
}

double LidDrivenCavity::GetLx() {
    return Lx;
}    

double LidDrivenCavity::GetLy() {
    return Ly;
}    

double LidDrivenCavity::GetRe() {
    return Re;
}

double LidDrivenCavity::GetU() {
    return U;
}

double LidDrivenCavity::GetNu() {
    return nu;
}

void LidDrivenCavity::GetData(double* vOut, double* sOut, double* u0Out, double* u1Out) {
    for(int i = 0; i < Npts; ++i) {
        vOut[i] = v[i];              //copy data for vorticity and streamfunction
        sOut[i] = s[i];
    }
    
    //--------------------For checking velocity, code snippet exactly same as the one used in WriteSolution ---------------//
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0Out[IDX(i,j)] =  (sOut[IDX(i,j+1)] - sOut[IDX(i,j)]) / dy;     //compute velocity in x direction at every grid point from streamfunction
            u1Out[IDX(i,j)] = -(sOut[IDX(i+1,j)] - sOut[IDX(i,j)]) / dx;     //compute velocity in y direction at every grid point from streamfunction
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0Out[IDX(i,Ny-1)] = U;                                        //impose x velocity as U at top surface to enforce no-slip boundary condition
    }
    //----------------------------------------------------------------------------------------------------------------------//
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();                                                   //update grid spacing dx dy based off new domain
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    //update global values
    MPI_Allreduce(&Nx,&globalNx,1,MPI_INT,MPI_SUM,comm_row_grid);
    MPI_Allreduce(&Ny,&globalNy,1,MPI_INT,MPI_SUM,comm_col_grid);   //-> note for future, is there any point in splitting Lx Ly up in main, when I'm gonna need global values anyway?
    
    UpdateDxDy();                                                   //update grid spacing dx dy based off new number of grid points
}

void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;                                              //compute kinematic viscosity from Reynolds number
}

void LidDrivenCavity::Initialise()
{
    CleanUp();                                                      //deallocate memory

    v   = new double[Npts]();                                       //array denoting vorticity, allocated with zero initial condition
    s   = new double[Npts]();                                       //array denoting streamfunction, allocated with zero initial condition
    tmp = new double[Npts]();                                       //temporay array, zeros
    cg  = new SolverCG(Nx, Ny, dx, dy);                             //create solver
    
    vTopData = new double[Nx]();                   //top and bottom data have size local 1 x Nx
    vBottomData = new double[Nx]();
    vLeftData = new double[Ny]();                  //left and right data have size local Ny x 1
    vRightData = new double[Ny]();
    
    sTopData = new double[Nx]();                   //top and bottom data have size local 1 x Nx
    sBottomData = new double[Nx]();
    sLeftData = new double[Ny]();                  //left and right data have size local Ny x 1
    sRightData = new double[Ny]();
}

void LidDrivenCavity::Integrate()
{
    
    int NSteps = ceil(T/dt);                                        //number of time steps required, rounded up
    for (int t = 0; t < NSteps; ++t)
    {
        if((MPIcoords[0] == 0) & (MPIcoords[1] == 0)) {                           //only print on root rank
            std::cout << "Step: " << setw(8) << t
                      << "  Time: " << setw(8) << t*dt
                      << std::endl;                                     //after each step, output time and step information
        }
        Advance();                                                  //solve the spatial problem and the time domain problem for one time step
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    double* u0 = new double[Nx*Ny]();                               //u0 is horizontal x velocity, initialised with zeros
    double* u1 = new double[Nx*Ny]();                               //u1 is vertical y velocity, initialised with zeros
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;     //compute velocity in x direction at every grid point from streamfunction
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;     //compute velocity in y direction at every grid point from streamfunction
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;                                        //impose x velocity as U at top surface to enforce no-slip boundary condition
    }

    std::ofstream f(file.c_str());                                  //open/create file for output
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)                                //print data in columns (i.e.keep x location constant, and go down y location)
        {
            k = IDX(i, j);                                                  //denotes location of matrix element (i,j) in memory
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k]     //on each line in file, print the grid location (x,y), vorticity...
              << " " << u0[k] << " " << u1[k] << std::endl;                 //streamfunction, x velocity, y velocity at that grid location
        }
        f << std::endl;                                                     //After printing all (y) data for column in grid, proceed to next column...
    }                                                                       //with a space to differentiate between each column
    f.close();                                                      //close file

    delete[] u0;                                                    //deallocate memory
    delete[] u1;
}

void LidDrivenCavity::PrintConfiguration()
{
    if((MPIcoords[0] == 0) & (MPIcoords[1]== 0)) {
        cout << "Grid size: " << globalNx << " x " << globalNy << endl;                         //print the current problem configuration
        cout << "Spacing:   " << dx << " x " << dy << endl;
        cout << "Length:    " << Lx << " x " << Ly << endl;
        cout << "Grid pts:  " << globalNx*globalNy << endl;
        cout << "Timestep:  " << dt << endl;
        cout << "Steps:     " << ceil(T/dt) << endl;
        cout << "Reynolds number: " << Re << endl;
        cout << "Linear solver: preconditioned conjugate gradient" << endl;
        cout << endl;
    }
    
    if (nu * dt / dx / dy > 0.25) {                                             //if timestep restriction not satisfied, terminate the program
        if((MPIcoords[0] == 0) & (MPIcoords[1] == 0)) {                     //only print on root tank
            cout << "ERROR: Time-step restriction not satisfied!" << endl;
            cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        }
        exit(-1);
    }
}

void LidDrivenCavity::CleanUp()
{
    if (v) {                //if array v is not null pointer, then deallocate arrays and solverCG 
        delete[] v;
        delete[] s;
        delete[] tmp;
        delete cg;
        
        delete[] vTopData;
        delete[] vBottomData;
        delete[] vRightData;
        delete[] vLeftData;
        
        delete[] sTopData;
        delete[] sBottomData;
        delete[] sRightData;
        delete[] sLeftData;
    }
}

void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (globalNx-1);       //calculate new spatial steps dx and dy based off current grid numbers (Nx,Ny) and domain size (Lx,Ly)
    dy = Ly / (globalNy-1);
    
    Npts = Nx * Ny;         //total number of grid points, locally
}

void LidDrivenCavity::Advance()
{
    //----------------Send streamfunction data asap -----------------// Note: maybe send left right data first, as no processing
    //if not top row, send data from bottom row of current process to process in grid below so that grid below now has top data for five point stencil
    //similar logic for bottom, left and right
    //for rows, need to extract data first
    cblas_dcopy(Nx, s, Ny, vTopData, 1);    //store only top row of streamfunction data into vTopData, Nx data to send upwards
    cblas_dcopy(Nx, s+Ny-1, Ny, vBottomData, 1);    //store only bottom row of streamfunction data into vBottomData to be sent, Nx data to send
    
    if(topRank != MPI_PROC_NULL) {               //send data upwards unless at teh top boundary
        MPI_Send(vTopData, Nx, MPI_DOUBLE, topRank, 0, comm_col_grid);                  //tag = 0 -> streamfunction data sent up
    }
    if(bottomRank != MPI_PROC_NULL){                   //send data downwards unlsss at teh bottom bounday
        MPI_Send(vBottomData, Nx, MPI_DOUBLE, bottomRank, 1, comm_col_grid);            //tag = 1 -> streamfunction data sent down
    }
    
    //for left and right columns, no need to extract data first as already in column major
    //send data with Ny datapoints and s+Ny*(Nx-1) denotes start of last column (i.e right most column)
    if(leftRank != MPI_PROC_NULL) {                //send data left unless at the left boundary
        MPI_Send(s+Ny*(Nx-1),Ny,MPI_DOUBLE,leftRank, 2, comm_row_grid);             //tag = 2 -> streamfunction data sent left
    }
    //s denotes start of leftmost column
    if(rightRank != MPI_PROC_NULL) {                //send data right unless at the right boundary
        MPI_Send(s,Ny,MPI_DOUBLE,rightRank,3,comm_row_grid);                        //tag = 3 -> streamfunction data sent right
    }
    
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;                                                //store 1/dx,1/dy,1/dx/dx,1/dy/dy to optimise performance

    //receive top and bottom streamfunction data first
    if(topRank != MPI_PROC_NULL) {               //all ranks but the top will receive streamfunction data from above, tag 1
        MPI_Recv(sTopData,Nx,MPI_DOUBLE,topRank,1,comm_col_grid,MPI_STATUS_IGNORE);
    }
    if(bottomRank != MPI_PROC_NULL) {                  //all ranks but the bottom will receive streamfunction data from below, tag 0
        MPI_Recv(sBottomData,Nx,MPI_DOUBLE,bottomRank,0,comm_col_grid,MPI_STATUS_IGNORE);
    }
    
    // Boundary node vorticity, for edge case where only one data row or column, so need to access data from other processes
    
    //break up BC assignment to top bottom left right separately, due to gridded nature
    if(bottomRank == MPI_PROC_NULL) {          //assign bottom BC
        if(Ny == 1) {       //first capture edge case where only one row, so second row in process above
            for(int i = 1; i < Nx - 1; ++i) {
                v[IDX(i,0)]     = 2.0 * dy2i * (s[IDX(i,0)]   - sTopData[i]);
            }
        }
        else {              //for more general case 
            for(int i = 1; i < Nx-1; ++i) {
                v[IDX(i,0)] = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
            }
        }
    }
    
    if(topRank == MPI_PROC_NULL) {              //assign top BC
        if(Ny == 1) {           //first capture edge case where only one row, so second row in process below
            for(int i = 1; i < Nx - 1; ++i) {
                v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - sBottomData[i]) - 2.0 * dyi * U;
            }
        }
        else {      //more general case
            for(int i = 1; i < Nx - 1; ++i) {
                v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)]) - 2.0 * dyi * U;
            }
        }
    }
    
    //now receive left and right data
    if(leftRank != MPI_PROC_NULL) {               //all ranks but the left will receive streamfunction data from left, tag 3
        MPI_Recv(sLeftData,Ny,MPI_DOUBLE,leftRank,3,comm_row_grid,MPI_STATUS_IGNORE);
    }
    if(rightRank != MPI_PROC_NULL) {               //all ranks but the right will receive streamfunction data from right, tag 2
        MPI_Recv(sRightData,Ny,MPI_DOUBLE,rightRank,2,comm_row_grid,MPI_STATUS_IGNORE);
    }
    
    //impose left right BCs
    if(leftRank == MPI_PROC_NULL) {              //assign left BC
        if(Nx == 1) {           //first capture edge case where only one column, so second column in process to the right
            for(int j = 1; j < Ny - 1; ++j) {
                v[IDX(0,j)] = 2.0 * dx2i * (s[IDX(0,j)] - sRightData[j]);
            }
        }
        else {      //more general case
            for(int j = 1; j < Nx - 1; ++j) {
                v[IDX(0,j)] = 2.0 * dx2i * (s[IDX(0,j)] - s[IDX(1,j)]);
            }
        }
    }
    
    if(rightRank == MPI_PROC_NULL) {              //assign right BC
        if(Nx == 1) {           //first capture edge case where only one column, so second column in process to the left
            for(int j = 1; j < Ny - 1; ++j) {
                v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - sLeftData[j]);
            }
        }
        else {      //more general case
            for(int j = 1; j < Nx - 1; ++j) {
                v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);
            }
        }
    }

    // Compute interior vorticity
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            v[IDX(i,j)] = dx2i*(
                    2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])                  //relating to x terms
                        + 1.0/dy/dy*(
                    2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);                 //relating to y terms
        }
    }

    // Time advance vorticity
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            v[IDX(i,j)] = v[IDX(i,j)] + dt*(
                ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
                 *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
                 *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
              + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    exit(-1);
    // Solve Poisson problem
    cg->Solve(v, s);
}