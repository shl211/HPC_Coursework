#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

/**
 * @brief Macro to map coordinates (i,j) onto it's corresponding location in memory, assuming row-wise matrix storage
 * @param I     coordinate i denoting horizontal position of grid from left to right
 * @param J     coordinate j denoting vertical position of grid from bottom to top
 */
#define IDX(I,J) ((J)*Nx + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

LidDrivenCavity::LidDrivenCavity()
{
    //create Cartesian communicator and row and column communicators, also assigns size of row/column communicators
    CreateCartGrid(comm_Cart_grid,comm_row_grid,comm_col_grid);
    
    //compute ranks along the row communciator and the column communicator
    MPI_Comm_rank(comm_row_grid, &rowRank);                             
    MPI_Comm_rank(comm_col_grid, &colRank);

    //compute ranks for adjacent grids for data transfer, if at boundary, returns -2 (MPI_PROC_NULL)
    MPI_Cart_shift(comm_col_grid,0,1,&bottomRank,&topRank);
    MPI_Cart_shift(comm_row_grid,0,1,&leftRank,&rightRank);
    
    if((topRank != MPI_PROC_NULL) && (bottomRank != MPI_PROC_NULL) && (leftRank != MPI_PROC_NULL) && (rightRank != MPI_PROC_NULL))
        boundaryDomain = false;
    else
        boundaryDomain = true;                                      //check whether the current process is on the edge of the global grid domain

    //first discretise the default domain into grids, in unlikely case default case is used
    globalNx = Nx;
    globalNy = Ny;
    globalLx = Lx;
    globalLy = Ly;                                                  //assign global values

    //now discretise grid domains and update values appropriately
    SplitDomainMPI(comm_Cart_grid,globalNx,globalNy,globalLx, globalLy, Nx, Ny, Lx, Ly,xDomainStart,yDomainStart);
    UpdateDxDy();
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();                                                      //deallocate memory

    //clean up MPI communicators
    MPI_Comm_free(&comm_Cart_grid);
    MPI_Comm_free(&comm_row_grid);
    MPI_Comm_free(&comm_col_grid);
}

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
    return Nx;
}

int LidDrivenCavity::GetNy() {
    return Ny;
}

int LidDrivenCavity::GetNpts() {
    return Nx*Ny;
}

int LidDrivenCavity::GetGlobalNpts() {
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

int LidDrivenCavity::GetGlobalNx(){
    return globalNx;
}

int LidDrivenCavity::GetGlobalNy(){
    return globalNy;
}

double LidDrivenCavity::GetGlobalLx(){
    return globalLx;
}

double LidDrivenCavity::GetGlobalLy(){
    return globalLy;
}

void LidDrivenCavity::GetData(double* vOut, double* sOut) {
    
    //correct array size is assumed
    cblas_dcopy(Npts,v,1,vOut,1);
    cblas_dcopy(Npts,s,1,sOut,1);
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    //global values are entered and stored
    globalLx = xlen;
    globalLy = ylen;

    //split domain up appropriately and update local values, including grid spacing
    SplitDomainMPI(comm_Cart_grid, globalNx, globalNy, globalLx, globalLy,Nx, Ny, Lx,Ly,xDomainStart,yDomainStart);
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    //global values are entered and stored
    globalNx = nx;
    globalNy = ny;

    //split domain up appropriately and update local values
    SplitDomainMPI(comm_Cart_grid, globalNx, globalNy, globalLx, globalLy,Nx, Ny, Lx,Ly,xDomainStart,yDomainStart);
    UpdateDxDy();
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
     //compute kinematic viscosity from Reynolds number
    this->Re = re;
    this->nu = 1.0/re;
}

void LidDrivenCavity::Initialise()
{
    CleanUp();                                                           //deallocate memory

    //allocate member variable arrays and initialise with zeros
    v   = new double[Npts]();
    vNext = new double[Npts]();
    s   = new double[Npts]();
    tmp = new double[Npts]();
    cg  = new SolverCG(Nx, Ny, dx, dy,comm_row_grid,comm_col_grid);
    
    vTopData = new double[Nx]();                                        //top and bottom data row have size local 1 x Nx
    vBottomData = new double[Nx]();
    vLeftData = new double[Ny]();                                       //left and right data column have size local Ny x 1
    vRightData = new double[Ny]();
    
    sTopData = new double[Nx]();
    sBottomData = new double[Nx]();
    sLeftData = new double[Ny]();
    sRightData = new double[Ny]();

    //no need to initalise these temporary arrays, will be overwritten
    tempLeft = new double[Ny];
    tempRight = new double[Ny];
}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);                                        //number of time steps required, rounded up
    for (int t = 0; t < NSteps; ++t)
    {
        if((rowRank == 0) && (colRank == 0)) {                       //only print on root rank
            std::cout << "Step: " << setw(8) << t
                      << "  Time: " << setw(8) << t*dt
                      << std::endl;                                 //after each step, output time and step information
        }
        Advance();                                                  //compute flow properties across domain for next time step
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    //compute velocities locally before sending -> faster than gathering then calculating
    double* u0 = new double[Nx*Ny]();                                                   //u0 is horizontal x velocity
    double* u1 = new double[Nx*Ny]();                                                   //u1 is vertical y velocity

    ComputeVelocity(u0,u1);

    //------------------------------------------Gather Data to Write Solution to File--------------------------------------------------------------//
    /*Data stored in row major format and printed columnwise. Gather all data at root of each column communicator
    Root column process (bottom row of grid) holds all data for the entire column, can print sequentially from left to right
    Root column processes have rank colRank = 0 and share a row communicator (this exploits sequential labelling of ranks in Cartesian subgrids row and columns)*/

    double* sAllCol = new double[Nx*globalNy]();
    double* vAllCol = new double[Nx*globalNy]();

    double* u0AllCol = new double[Nx*globalNy]();
    double* u1AllCol = new double[Nx*globalNy]();         

    //using GatherV as each process holds different number of data
    int* colRecDataNum = new int[size];         //how many data points to be received from each process in column communicator
    int* relativeDisp = new int[size];          //where data should be stored relative to send buffer pointer
    int rel = yDomainStart*Nx;                  //where current process data would go in the column communicator gathered matrix

    MPI_Gather(&Npts,1,MPI_INT,colRecDataNum+colRank,1,MPI_INT,0,comm_col_grid);        //root needs this info for Gatherv
    MPI_Gather(&rel,1,MPI_INT,relativeDisp+colRank,1,MPI_INT,0,comm_col_grid);

    //send local data for s and v of each process to correct place in root column; AllCol now data for the entire column communicator
    MPI_Gatherv(s,Npts,MPI_DOUBLE,sAllCol,colRecDataNum,relativeDisp,MPI_DOUBLE,0,comm_col_grid);       
    MPI_Gatherv(vNext,Npts,MPI_DOUBLE,vAllCol,colRecDataNum,relativeDisp,MPI_DOUBLE,0,comm_col_grid);       
    MPI_Gatherv(u0,Npts,MPI_DOUBLE,u0AllCol,colRecDataNum,relativeDisp,MPI_DOUBLE,0,comm_col_grid);       
    MPI_Gatherv(u1,Npts,MPI_DOUBLE,u1AllCol,colRecDataNum,relativeDisp,MPI_DOUBLE,0,comm_col_grid);   

    //only root column ranks can write to file
    if(colRank == 0) {
        std::ofstream f;
        int goAheadMessage = 0;                                         //Receive buffer that will be used to end a blocking receive on the adjacent column rank

        //write left column first (rowRank = 0) to right (row communnicator ranks ordered from left to right)
        if(rowRank == 0) {
            //for row rank 0 (prints data first) open file to overwrite
            f.open(file.c_str(),std::ios::trunc);
            std::cout << "Writing file " << file << std::endl;
        }
        else{
            //blocking receive to force process to wait until LHS root column process finished writing before writing -> ensure column-by-column format
            MPI_Recv(&goAheadMessage,1,MPI_INT,leftRank,10,comm_row_grid,MPI_STATUS_IGNORE);
            f.open(file.c_str(),std::ios::app);                         //other processes should append data to file, not overwrite
        }
        
        int k = 0;
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < globalNy; ++j)                                  //print data in columns
            {
                k = IDX(i, j);
                f << (i + xDomainStart) * dx << " " << (j + yDomainStart) * dy  //i+xDomainStart accounts for where local column starts in the global x direction
                << " " << vAllCol[k] <<  " " << sAllCol[k]                      //on each line in file, print the grid location (x,y), vorticity...
                << " " << u0AllCol[k] << " " << u1AllCol[k] << std::endl;       //streamfunction, x velocity, y velocity at that grid location
            }
            f << std::endl;                                                     //After printing all (y) data for column in grid, proceed to next column...
        }                                                                       //with a space to differentiate between each column
        f.close();

        //writing done for this process, tell next process to go by sending a message to unblock the next root column process
        MPI_Send(&goAheadMessage,1,MPI_INT,rightRank,10,comm_row_grid);
    }

    delete[] u0;
    delete[] u1;
    delete[] sAllCol;
    delete[] vAllCol;
    delete[] u0AllCol;
    delete[] u1AllCol;
    //ensure all processes have finished writing before proceeding, prevents access errors if file to be opened after end of function
    MPI_Barrier(MPI_COMM_WORLD);                                                
}

void LidDrivenCavity::PrintConfiguration()
{
    if((rowRank == 0) && (colRank == 0)) {                                       //only print on root rank
        cout << "Grid size: " << globalNx << " x " << globalNy << endl;         //print the current global problem configuration of the lid driven cavity
        cout << "Spacing:   " << dx << " x " << dy << endl;
        cout << "Length:    " << globalLx << " x " << globalLy << endl;
        cout << "Grid pts:  " << globalNx*globalNy << endl;
        cout << "Timestep:  " << dt << endl;
        cout << "Steps:     " << ceil(T/dt) << endl;
        cout << "Reynolds number: " << Re << endl;
        cout << "Linear solver: preconditioned conjugate gradient" << endl;
        cout << endl;
    }
    
    if (nu * dt / dx / dy > 0.25) {                                             //if timestep restriction not satisfied, terminate the program
        if((rowRank == 0) && (colRank == 0)) {
            cout << "ERROR: Time-step restriction not satisfied!" << endl;
            cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        }

        MPI_Finalize();
        exit(-1);
    }
}

void LidDrivenCavity::CleanUp()
{
    //if arrays are allocated, deallocate
    if (v) {                        
        delete[] v;
        delete[] vNext;
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

        delete[] tempLeft;
        delete[] tempRight;
    }
}

void LidDrivenCavity::UpdateDxDy()
{
    //calculate new spatial steps dx and dy based off current global grid numbers (Nx,Ny) and domain size (Lx,Ly)
    dx = globalLx / (globalNx-1);       
    dy = globalLy / (globalNy-1);
    
    Npts = Nx * Ny;                 //total number of local grid points
}

void LidDrivenCavity::Advance()
{
    //compute current vorticity from streamfunction with 2nd order finite central difference (2FCD)
    ComputeVorticity();

    //compute vorticity at next time step from current time step with streamfunction adn vorticity with 2FCD
    ComputeTimeAdvanceVorticity();

    // Solve Poisson problem to get streamfunction at next time step -> flow properties at next time step now known
    cg->Solve(vNext, s);
}

void LidDrivenCavity::ComputeVorticity() {

    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;    

    //---------------------------------------------------------------------------------------------------------------------------//
    //------------------------------------Step 1: Transfer Data and Compute Interior Points--------------------------------------//
    //---------------------------------------------------------------------------------------------------------------------------//

    //send streamfunction boundary data in all directions
    MPI_Isend(s+Nx*(Ny-1), Nx, MPI_DOUBLE, topRank, 0, comm_col_grid,&requests[0]);                 //tag = 0 -> streamfunction data sent up
    MPI_Isend(s, Nx, MPI_DOUBLE, bottomRank, 1, comm_col_grid,&requests[1]);                        //tag = 1 -> streamfunction data sent down
    
    //extract and send left and right
    cblas_dcopy(Ny,s,Nx,tempLeft,1);
    cblas_dcopy(Ny,s+Nx-1,Nx,tempRight,1);
    MPI_Isend(tempLeft,Ny,MPI_DOUBLE,leftRank, 2, comm_row_grid,&requests[2]);                      //tag = 2 -> streamfunction data sent left
    MPI_Isend(tempRight,Ny,MPI_DOUBLE,rightRank,3,comm_row_grid,&requests[3]);                      //tag = 3 -> streamfunction data sent right

    //compute interior vorticity points while waiting for data to send
    //dynamic scheduling observed in tests to be better for load balancing
    #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                v[IDX(i,j)] = dx2i*( 2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])
                            + dy2i*( 2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);
            }
        }

    //receive boundary data
    MPI_Recv(sTopData,Nx,MPI_DOUBLE,topRank,1,comm_col_grid,MPI_STATUS_IGNORE);                     //bottom row of process is data sent up from process below              
    MPI_Recv(sBottomData,Nx,MPI_DOUBLE,bottomRank,0,comm_col_grid,MPI_STATUS_IGNORE);               //top row of process is data send down from process above
    MPI_Recv(sLeftData,Ny,MPI_DOUBLE,leftRank,3,comm_row_grid,MPI_STATUS_IGNORE);                   //right column of process is data sent from process to right
    MPI_Recv(sRightData,Ny,MPI_DOUBLE,rightRank,2,comm_row_grid,MPI_STATUS_IGNORE);                 //left column of process is data sent from process to left

    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 2: Compute Vorticity on Corners of Local Domain------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//

    //if single cell domain not on global boundary, need acess to all four datasets
    if((Nx == 1) && (Ny == 1) && !boundaryDomain) {   
        v[0] = dx2i * (2.0 * s[0] - sRightData[0] - sLeftData[0])
            + dy2i * (2.0 * s[0] - sTopData[0] - sBottomData[0]);
    }
    //unless at global left/right where BC is imposed, if local domain is column vector, do:
    else if ((Nx == 1) && (Ny != 1)  && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
        //if it's top boundary process, don't repeat calculation of top 'corner'; same logic for bottom
        if(topRank != MPI_PROC_NULL) { 
            v[IDX(0,Ny-1)] = dx2i * (2.0 * s[IDX(0,Ny-1)] - sRightData[0] - sLeftData[0])
                        + dy2i * (2.0 * s[IDX(0,Ny-1)] - sTopData[0] - s[IDX(0,Ny-2)]);              
        }

        if(bottomRank != MPI_PROC_NULL) {
            v[IDX(0,0)] = dx2i * (2.0 * s[IDX(0,0)] - sRightData[0] - sLeftData[0])
                        + dy2i * (2.0 * s[IDX(0,0)] - s[IDX(0,1)] - sBottomData[0]);
        }
    }
    //unless at global top/bottom where BC is imposed, if local domain is row vector, do:
    else if ((Nx != 1) && (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {    
        //if it's left boundary process, don't repeat calculation of left 'corner'; same logic for right 
        if(leftRank != MPI_PROC_NULL) { 
            v[IDX(0,0)] = dx2i * (2.0 * s[IDX(0,0)] - s[IDX(1,0)] - sLeftData[0])
                        + dy2i * (2.0 * s[IDX(0,0)] - sTopData[0] - sBottomData[0]);
        }

        if(rightRank != MPI_PROC_NULL) { 
            v[IDX(Nx-1,0)] = dx2i * (2.0 * s[IDX(Nx-1,0)] - sRightData[0] - s[IDX(Nx-2,0)])
                        + dy2i * (2.0 * s[IDX(Nx-1,0)] - sTopData[Nx-1] - sBottomData[Nx-1]);   
        }
    }
    //compute corners for general case
    else {
        //don't repeat calculation for bottom left corner of process domain if process is at the left or bottom of grid (BC already imposed)
        //same logic for bottom right, top left, and top right
        if(!((bottomRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            v[IDX(0,0)] = dx2i * (2.0 * s[IDX(0,0)] - s[IDX(1,0)] - sLeftData[0])
                            + dy2i * (2.0 * s[IDX(0,0)] - s[IDX(0,1)] - sBottomData[0]);  
        }
        
        if(!((bottomRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
            v[IDX(Nx-1,0)] = dx2i * (2.0 * s[IDX(Nx-1,0)] - sRightData[0] - s[IDX(Nx-2,0)])
                        + dy2i * (2.0 * s[IDX(Nx-1,0)] - s[IDX(Nx-1,1)] - sBottomData[Nx-1]);
        }
        
        if(!((topRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            v[IDX(0,Ny-1)] = dx2i * (2.0 * s[IDX(0,Ny-1)] - s[IDX(1,Ny-1)] - sLeftData[Ny-1]) 
                        + dy2i * (2.0 * s[IDX(0,Ny-1)] - sTopData[0] - s[IDX(0,Ny-2)]);
        }
        
        if(!((topRank == MPI_PROC_NULL )|| (rightRank == MPI_PROC_NULL))) {
            v[IDX(Nx-1,Ny-1)] = dx2i * (2.0 * s[IDX(Nx-1,Ny-1)] - sRightData[Ny-1] - s[IDX(Nx-2,Ny-1)])
                        + dy2i * (2.0 * s[IDX(Nx-1,Ny-1)] - sTopData[Nx-1] - s[IDX(Nx-1,Ny-2)]);
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 3: Compute Vorticity on Edges of Local Domain--------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//

    //unless at global left/right where BC is imposed, if local domain is column vector, compute data between top and bottom corners
    if((Nx == 1) && (Ny > 1 ) && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {    
        for(int j = 1; j < Ny - 1; ++j) {
            v[IDX(0,j)] = dx2i * (2.0 * s[IDX(0,j)] - sRightData[j] - sLeftData[j])
                        + dy2i * (2.0 * s[IDX(0,j)] - s[IDX(0,j+1)] - s[IDX(0,j-1)]);
        }
    }
    //unless at global top/bottom where BC is imposed, if local domain is row vector, compute data between left and right corners
    else if((Nx > 1) && (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {
        for(int i = 1; i < Nx - 1; ++i) {
            v[IDX(i,0)] =  dx2i * (2.0 * s[IDX(i,0)] - s[IDX(i+1,0)] - s[IDX(i-1,0)])
                        + dy2i * (2.0 * s[IDX(i,0)] - sTopData[i] - sBottomData[i]);
        }
    }
    //compute edges for general case (exclude single cell domain -> 'corner' is sufficient)
    else if((Nx != 1) && (Ny != 1)) {
        //if process at bottom of grid, don't need to do anything as BC imposed
        //same logic for other sides
        if(bottomRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                v[IDX(i,0)] =  dx2i * (2.0 * s[IDX(i,0)] - s[IDX(i+1,0)] - s[IDX(i-1,0)])
                            + dy2i * (2.0 * s[IDX(i,0)] - s[IDX(i,1)] - sBottomData[i]);
            }
        }

        if(topRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                v[IDX(i,Ny-1)] = dx2i * (2.0 * s[IDX(i,Ny-1)] - s[IDX(i+1,Ny-1)] - s[IDX(i-1,Ny-1)])
                            + dy2i * (2.0 * s[IDX(i,Ny-1)] - sTopData[i] - s[IDX(i,Ny-2)]);
            }
        }

        if(leftRank != MPI_PROC_NULL) {
            for(int j = 1; j < Ny - 1; ++j) {
                v[IDX(0,j)] = dx2i * (2.0 * s[IDX(0,j)] - s[IDX(1,j)] - sLeftData[j])
                            + dy2i * (2.0 * s[IDX(0,j)] - s[IDX(0,j+1)] - s[IDX(0,j-1)]);
            }
        }

        if(rightRank != MPI_PROC_NULL) {        
            for(int j = 1; j < Ny - 1; ++j) {
                v[IDX(Nx-1,j)] = dx2i * (2.0 * s[IDX(Nx-1,j)] - sRightData[j] - s[IDX(Nx-2,j)])
                            + dy2i * (2.0 * s[IDX(Nx-1,j)] - s[IDX(Nx-1,j+1)] - s[IDX(Nx-1,j-1)]);
            }
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 4: Impose Global Boundary Conditions-----------------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//    
    //no parallel region here as testing with Lx,Ly=1, Nx,Ny=201,Re=1000,dt=0.005,T-0.1 always led to slower performance//
    //note that corners are left untouched as per original code

    //some data stored here to simplify if statements for identifying global corners
    double val1,val2;     

    //assign bottom BC, only special case is row vector (single cell is subset)
    if(bottomRank == MPI_PROC_NULL) {                     
        //if domain is row vector, need data from process above, unless at global left/right where BC is imposed
        if( (Ny == 1) && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL)) ) {
            for(int i = 1; i < Nx - 1; ++i)
                v[IDX(i,0)]     = 2.0 * dy2i * (s[IDX(i,0)]   - sTopData[i]);

            //store these two values to reduce conditional statements; not done for loops as here only two values to copy
            val1 = sTopData[0];            //leftmost point from top process
            val2 = sTopData[Nx-1];         //rightmost point from top process
        }
        //otherwise for general case at bottom of grid, impose bottom BC -> consider case of bottom left and right corners
        else{
            //otherwise, for general case at bottom of grid, impose these bottom BCs 
            for(int i = 1; i < Nx-1; ++i)
                v[IDX(i,0)] = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);

            val1 = s[IDX(0,1)];
            val2 = s[IDX(Nx-1,1)];
        }
        
        //if not bottom left global grid corner, also compute bottom left corner
        if(leftRank != MPI_PROC_NULL) 
            v[IDX(0,0)] = 2.0 * dy2i * (s[IDX(0,0)] - val1);
                
        //if not top bottom global grid corner, also compute bottom right corner
        if(rightRank != MPI_PROC_NULL)
            v[IDX(Nx-1,0)] = 2.0 * dy2i * (s[IDX(Nx-1,0)] - val2);
    }
    
    //assign top BC, same logic as bottom BCs
    if(topRank == MPI_PROC_NULL) {              
        if((Ny == 1) && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL)))
        {
            for(int i = 1; i < Nx - 1; ++i)
                v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - sBottomData[i]) - 2.0 * dyi * U;

            val1 = sBottomData[0];            //leftmost point from bottom process
            val2 = sBottomData[Nx-1];         //rightmost point from bottom process
        }
        else {
            for(int i = 1; i < Nx - 1; ++i)
                v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)]) - 2.0 * dyi * U;

            val1 = s[IDX(0,Ny-2)];
            val2 = s[IDX(Nx-1,Ny-2)];
        }

    if(leftRank != MPI_PROC_NULL)
        v[IDX(0,Ny-1)] = 2.0 * dy2i * (s[IDX(0,Ny-1)] - val1) - 2.0 * dyi * U;
        
    if(rightRank != MPI_PROC_NULL)
        v[IDX(Nx-1,Ny-1)] = 2.0 * dy2i * (s[IDX(Nx-1,Ny-1)] - val2) - 2.0 * dyi * U;
    }
    
    //assign left BC, only special case is column vector
    if(leftRank == MPI_PROC_NULL) {              
        //if domain is column vector, need data from process to right, unless at global top/bottom where BC is imposed
        if((Nx == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {
            for(int j = 1; j < Ny - 1; ++j)
                v[IDX(0,j)] = 2.0 * dx2i * (s[IDX(0,j)] - sRightData[j]);

            val1 = sRightData[Ny-1];        //top point of right process
            val2 = sRightData[0];           //bottom point of right process
        }
        else{
            //otherwise, for general case at left of grid, impose these left BCs 
            for(int j = 1; j < Ny - 1; ++j)
                v[IDX(0,j)] = 2.0 * dx2i * (s[IDX(0,j)] - s[IDX(1,j)]);

            val1 = s[IDX(1,Ny-1)];
            val2 = s[IDX(1,0)];
        }

        //if not top left process, also compute top left corner
        if(topRank != MPI_PROC_NULL)
            v[IDX(0,Ny-1)] = 2.0 * dx2i * (s[IDX(0,Ny-1)] - val1);

        //if not bottom left process, also compute bottom left corner
        if(bottomRank != MPI_PROC_NULL)
            v[IDX(0,0)] = 2.0 * dx2i * (s[IDX(0,0)] - val2);
    }

    //assign right BC, same logic as left
    if(rightRank == MPI_PROC_NULL) {              
        if((Nx == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {
            for(int j = 1; j < Ny - 1; ++j)
                v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - sLeftData[j]);

            val1 = sLeftData[Ny-1];        //top point of left process
            val2 = sLeftData[0];           //bottom point of left process
        }
        else {
            for(int j = 1; j < Ny - 1; ++j)
                v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);

            val1 = s[IDX(Nx-2,Ny-1)];
            val2 = s[IDX(Nx-2,0)];
        }

        if(topRank != MPI_PROC_NULL)
            v[IDX(Nx-1,Ny-1)] = 2.0 * dx2i * (s[IDX(Nx-1,Ny-1)] - val1);
    
        if(bottomRank != MPI_PROC_NULL)
            v[IDX(Nx-1,0)] = 2.0 * dx2i * (s[IDX(Nx-1,0)] - val2);
    }

    //ensure all communications completed
    MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);   
}

void LidDrivenCavity::ComputeTimeAdvanceVorticity() {
    //assume s data already sent and received by ComputeVorticity
    double dxi  = 1.0/dx;           //precompute some common division terms
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;

    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 1: Transfer Data and Compute Interior Points---------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//

    //send vorticity data on edge of each domain to adjacent grid
    MPI_Isend(v+Nx*(Ny-1), Nx, MPI_DOUBLE, topRank, 0, comm_col_grid,&requests[0]);     //tag = 0 -> streamfunction data sent up
    MPI_Isend(v, Nx, MPI_DOUBLE, bottomRank, 1, comm_col_grid,&requests[1]);            //tag = 1 -> streamfunction data sent down
    
    cblas_dcopy(Ny,v,Nx,tempLeft,1);                                                    //extract left and right data to be sent
    cblas_dcopy(Ny,v+Nx-1,Nx,tempRight,1);

    MPI_Isend(tempLeft,Ny,MPI_DOUBLE,leftRank, 2, comm_row_grid,&requests[2]);           //tag = 2 -> streamfunction data sent left
    MPI_Isend(tempRight,Ny,MPI_DOUBLE,rightRank,3,comm_row_grid,&requests[3]);          //tag = 3 -> streamfunction data sent right
    
    //compute interior points of v_n+1 to allow all data to be sent; requires only data stored in current process
    #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                vNext[IDX(i,j)] = v[IDX(i,j)] + dt*(
                        ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
                        *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
                    - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
                        *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
                    + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
                    + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
            }
        }
    
    //receive the data as need it for next process
    MPI_Recv(vTopData,Nx,MPI_DOUBLE,topRank,1,comm_col_grid,MPI_STATUS_IGNORE);                     //bottom row of process is data sent up from process below              
    MPI_Recv(vBottomData,Nx,MPI_DOUBLE,bottomRank,0,comm_col_grid,MPI_STATUS_IGNORE);               //top row of process is data send down from process above
    MPI_Recv(vLeftData,Ny,MPI_DOUBLE,leftRank,3,comm_row_grid,MPI_STATUS_IGNORE);                   //right column of process is data sent from process to right
    MPI_Recv(vRightData,Ny,MPI_DOUBLE,rightRank,2,comm_row_grid,MPI_STATUS_IGNORE);                 //left column of process is data sent from process to left

    //------------------------------------------------------------------------------------------------------------------------------------//
    //---------------------------------Step 2: Compute Time AdvanceVorticity on Corners of Local Domain-----------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//    
    if((Nx == 1) && (Ny == 1) && !boundaryDomain) {   
        //if domain effectively a cell, need data from all surrounding data
        vNext[0] = v[0] + dt * (
                ( (sRightData[0] - sLeftData[0]) * 0.5 * dxi
                 *(vTopData[0] - vBottomData[0]) * 0.5 * dyi)
               -( (sTopData[0] - sBottomData[0]) * 0.5 * dyi
                 *(vRightData[0] - vLeftData[0]) * 0.5 * dxi)
               + nu * (vRightData[0] - 2.0 * v[0] + vLeftData[0])*dx2i
               + nu * (vTopData[0] - 2.0 * v[0] + vBottomData[0])*dy2i);
    }        
    //unless at global left/right where BC is imposed, if local domain is column vector, do:
    else if ((Nx == 1) && (Ny != 1)  && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
        //if process is at top of grid, BC will be imposed, so skip 
        if(topRank != MPI_PROC_NULL) {  
            vNext[Ny-1] = v[Ny-1] + dt * (
                ( (sRightData[Ny-1] - sLeftData[Ny-1]) * 0.5 * dxi
                 *(vTopData[0] - v[Ny-2]) * 0.5 * dyi)
               -( (sTopData[0] - s[Ny-2]) * 0.5 * dyi
                 *(vRightData[Ny-1] - vLeftData[Ny-1]) * 0.5 * dxi)
               + nu * (vRightData[Ny-1] - 2.0 * v[0] + vLeftData[Ny-1])*dx2i
               + nu * (vTopData[0] - 2.0 * v[0] + v[Ny-2])*dy2i);                   //top 'corner'
        }    
        //same logic for bottom
        if(bottomRank != MPI_PROC_NULL) {   
            vNext[0] = v[0] + dt * (
                    ( (sRightData[0] - sLeftData[0]) * 0.5 * dxi
                        *(v[1] - vBottomData[0]) * 0.5 * dyi)
                    -( (s[1] - sBottomData[0]) * 0.5 * dyi
                        *(vRightData[0] - vLeftData[0]) * 0.5 * dxi)
                    + nu * (vRightData[0] - 2.0 * v[0] + vLeftData[0])*dx2i
                    + nu * (v[1] - 2.0 * v[0] + vBottomData[0])*dy2i);                   //bottom 'corner'
        }      
    }
    //unless at global top/bottom where BC is imposed, if local domain is row vector, do:
    else if ((Nx != 1) && (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {   
        //if process is at left of grid, BC will be imposed, so skip
        if(leftRank != MPI_PROC_NULL) {
            vNext[0] = v[0] + dt*(                                                  //left 'corner'
                        ( (s[1] - sLeftData[0]) * 0.5 * dxi
                         *(vTopData[0] - vBottomData[0]) * 0.5 * dyi)
                      - ( (sTopData[0] - sBottomData[0]) * 0.5 * dyi
                         *(v[1] - vLeftData[0]) * 0.5 * dxi)
                      + nu * (v[1] - 2.0 * v[0] + vLeftData[0])*dx2i
                      + nu * (vTopData[0] - 2.0 * v[0] + vBottomData[0])*dy2i);            
        }
        
        if(rightRank != MPI_PROC_NULL) {      
            //same logic for right
            vNext[Nx-1] = v[Nx-1] + dt*(                                            //right 'corner'
                ( (sRightData[0] - s[Nx-2]) * 0.5 * dxi
                 *(vTopData[Nx-1] - vBottomData[Nx-1]) * 0.5 * dyi)
              - ( (sTopData[Nx-1] - sBottomData[Nx-1]) * 0.5 * dyi
                 *(vRightData[0] - v[Nx-2]) * 0.5 * dxi)
              + nu * (vRightData[0] - 2.0 * v[Nx-1] + v[Nx-2])*dx2i
              + nu * (vTopData[Nx-1] - 2.0 * v[Nx-1] + vBottomData[Nx-1])*dy2i);
        }
    }
    else {
        if(!((bottomRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            vNext[IDX(0,0)] = v[IDX(0,0)] + dt*(                                    //compute bottom left corner, access left and bottom
                ( (s[IDX(1,0)] - sLeftData[0]) * 0.5 * dxi                          //if at left or bottom, BC will be imposed later
                 *(v[IDX(0,1)] - vBottomData[0]) * 0.5 * dyi)
                - ( (s[IDX(0,1)] - sBottomData[0]) * 0.5 * dyi
                *(v[IDX(1,0)] - vLeftData[0]) * 0.5 * dxi)
                + nu * (v[IDX(1,0)] - 2.0 * v[IDX(0,0)] + vLeftData[0])*dx2i
                + nu * (v[IDX(0,1)] - 2.0 * v[IDX(0,0)] + vBottomData[0])*dy2i);
        }
            
        if(!((bottomRank == MPI_PROC_NULL )|| (rightRank == MPI_PROC_NULL))) {
            vNext[IDX(Nx-1,0)] = v[IDX(Nx-1,0)] + dt*(                              //compute bottom right corner, acess right and bottom
                ( (sRightData[0] - s[IDX(Nx-2,0)]) * 0.5 * dxi                      //if at right or bottom, BC will be imposed later
                 *(v[IDX(Nx-1,1)] - vBottomData[Nx-1]) * 0.5 * dyi)
              - ( (s[IDX(Nx-1,1)] - sBottomData[Nx-1]) * 0.5 * dyi
                 *(vRightData[0] - v[IDX(Nx-2,0)]) * 0.5 * dxi)
              + nu * (vRightData[0] - 2.0 * v[IDX(Nx-1,0)] + v[IDX(Nx-2,0)])*dx2i
              + nu * (v[IDX(Nx-1,1)] - 2.0 * v[IDX(Nx-1,0)] + vBottomData[Nx-1])*dy2i);
        }
              
        if(!((topRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            vNext[IDX(0,Ny-1)] = v[IDX(0,Ny-1)] + dt*(                              //compute top left corner, access top and left
                ( (s[IDX(1,Ny-1)] - sLeftData[Ny-1]) * 0.5 * dxi                    //if at top or left, BC will be imposed later
                 *(vTopData[0] - v[IDX(0,Ny-2)]) * 0.5 * dyi)
              - ( (sTopData[0] - s[IDX(0,Ny-2)]) * 0.5 * dyi
                 *(v[IDX(1,Ny-1)] - vLeftData[Ny-1]) * 0.5 * dxi)
              + nu * (v[IDX(1,Ny-1)] - 2.0 * v[IDX(0,Ny-1)] + vLeftData[Ny-1])*dx2i
              + nu * (vTopData[0] - 2.0 * v[IDX(0,Ny-1)] + v[IDX(0,Ny-2)])*dy2i);
        }
              
        if(!((topRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
            vNext[IDX(Nx-1,Ny-1)] = v[IDX(Nx-1,Ny-1)] + dt*(                        //compute top right corner, access top and right
                ( (sRightData[Ny-1] - s[IDX(Nx-2,Ny-1)]) * 0.5 * dxi                //if at top or right, BC will be imposed later
                 *(vTopData[Nx-1] - v[IDX(Nx-1,Ny-2)]) * 0.5 * dyi)
              - ( (sTopData[Nx-1] - s[IDX(Nx-1,Ny-2)]) * 0.5 * dyi
                 *(vRightData[Ny-1] - v[IDX(Nx-2,Ny-1)]) * 0.5 * dxi)
              + nu * (vRightData[Ny-1] - 2.0 * v[IDX(Nx-1,Ny-1)] + v[IDX(Nx-2,Ny-1)])*dx2i
              + nu * (vTopData[Nx-1] - 2.0 * v[IDX(Nx-1,Ny-1)] + v[IDX(Nx-1,Ny-2)])*dy2i);
        }
    }
    
    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 2: Compute Vorticity on Edges of Local Domain--------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//    
    //no parallel region here as thread overheads exceed increase in speed of O(n) operations
    //tested with same benchmark, slowed things down when 'for' or 'sections' were introduced

    //if column vector, don't need to do for left or right as BC already imposed
    if((Nx == 1) && (Ny > 1) && !((leftRank == MPI_PROC_NULL )||( rightRank == MPI_PROC_NULL))) {
        for(int j = 1; j < Ny - 1; ++j) {
            vNext[j] = v[j] + dt*(                                                  //for column, only need left and right proecss data
                ( (sRightData[j] - sLeftData[j]) * 0.5 * dxi
                    *(v[j+1] - v[j-1]) * 0.5 * dyi)
                - ( (s[j+1] - s[j-1]) * 0.5 * dyi
                    *(vRightData[j] - vLeftData[j]) * 0.5 * dxi)
                + nu * (vRightData[j] - 2.0 * v[j] + vLeftData[j])*dx2i
                + nu * (v[j+1] - 2.0 * v[j] + v[j-1])*dy2i);
        }
    }
    else if ((Nx > 1 )&& (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {
        //if row vector, don't need to do for top or bottom as BC already imposed
        for(int i = 0; i < Nx - 1; ++i) {
            vNext[i] = v[i] + dt*(                                                  //row needs access to top an bototm
                ( (s[i+1] - s[i-1]) * 0.5 * dxi
                    *(vTopData[i] - vBottomData[i]) * 0.5 * dyi)
                - ( (sTopData[i] - sBottomData[i]) * 0.5 * dyi
                    *(v[i+1] - v[i-1]) * 0.5 * dxi)
                + nu * (v[i+1] - 2.0 * v[i] + v[i-1])*dx2i
                + nu * (vTopData[i] - 2.0 * v[i] + vBottomData[i])*dy2i);
        }
    }
    //for all other cases, process only needs to acces one dataset (exclude single domain as only needs 'corner')
    else if((Nx != 1) && (Ny != 1)) {
        //only compute bottom row between corners if not at bottom of grid
        if(bottomRank != MPI_PROC_NULL) {   
            for (int i = 1; i < Nx - 1; ++i) {                                      //bottom row, needs access to bottom
                vNext[IDX(i,0)] = v[IDX(i,0)] + dt*(
                        ( (s[IDX(i+1,0)] - s[IDX(i-1,0)]) * 0.5 * dxi
                            *(v[IDX(i,1)] - vBottomData[i]) * 0.5 * dyi)
                        - ( (s[IDX(i,1)] - sBottomData[i]) * 0.5 * dyi
                            *(v[IDX(i+1,0)] - v[IDX(i-1,0)]) * 0.5 * dxi)
                        + nu * (v[IDX(i+1,0)] - 2.0 * v[IDX(i,0)] + v[IDX(i-1,0)])*dx2i
                        + nu * (v[IDX(i,1)] - 2.0 * v[IDX(i,0)] + vBottomData[i])*dy2i);
            }
        }
            
        //only compute top row if not at top of grid
        if(topRank != MPI_PROC_NULL) {  
            for (int i = 1; i < Nx - 1; ++i) {                                      
                vNext[IDX(i,Ny-1)] = v[IDX(i,Ny-1)] + dt*(                          //top row, needs access to top
                        ( (s[IDX(i+1,Ny-1)] - s[IDX(i-1,Ny-1)]) * 0.5 * dxi
                            *(vTopData[i] - v[IDX(i,Ny-2)]) * 0.5 * dyi)
                        - ( (sTopData[i] - s[IDX(i,Ny-2)]) * 0.5 * dyi
                            *(v[IDX(i+1,Ny-1)] - v[IDX(i-1,Ny-1)]) * 0.5 * dxi)
                        + nu * (v[IDX(i+1,Ny-1)] - 2.0 * v[IDX(i,Ny-1)] + v[IDX(i-1,Ny-1)])*dx2i
                        + nu * (vTopData[i] - 2.0 * v[IDX(i,Ny-1)] + v[IDX(i,Ny-2)])*dy2i);
            }
        }
        
        //only compute left column if not at LHS of grid
        if(leftRank != MPI_PROC_NULL) {
            for (int j = 1; j < Ny - 1; ++j) {                                       //left column, needs access to left
                vNext[IDX(0,j)] = v[IDX(0,j)] + dt*(
                        ( (s[IDX(1,j)] - sLeftData[j]) * 0.5 * dxi
                            *(v[IDX(0,j+1)] - v[IDX(0,j-1)]) * 0.5 * dyi)
                        - ( (s[IDX(0,j+1)] - s[IDX(0,j-1)]) * 0.5 * dyi
                            *(v[IDX(1,j)] - vLeftData[j]) * 0.5 * dxi)
                        + nu * (v[IDX(1,j)] - 2.0 * v[IDX(0,j)] + vLeftData[j])*dx2i
                        + nu * (v[IDX(0,j+1)] - 2.0 * v[IDX(0,j)] + v[IDX(0,j-1)])*dy2i);
            }
        }
        
        //only compute right column if not at RHS of grid
        if(rightRank != MPI_PROC_NULL) {
            for (int j = 1; j < Ny - 1; ++j) {                                          
                vNext[IDX(Nx-1,j)] = v[IDX(Nx-1,j)] + dt*(                          //right column, needs access to right
                        ( (sRightData[j] - s[IDX(Nx-2,j)]) * 0.5 * dxi
                        *(v[IDX(Nx-1,j+1)] - v[IDX(Nx-1,j-1)]) * 0.5 * dyi)
                        - ( (s[IDX(Nx-1,j+1)] - s[IDX(Nx-1,j-1)]) * 0.5 * dyi
                        *(vRightData[j] - v[IDX(Nx-2,j)]) * 0.5 * dxi)
                        + nu * (vRightData[j] - 2.0 * v[IDX(Nx-1,j)] + v[IDX(Nx-2,j)])*dx2i
                        + nu * (v[IDX(Nx-1,j+1)] - 2.0 * v[IDX(Nx-1,j)] + v[IDX(Nx-1,j-1)])*dy2i);
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------------------------------------------//
    //-------------------------------------------------Step 4: Assign Global Boundary Conditions------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//
    
    if(bottomRank == MPI_PROC_NULL) {               //assign bottom BC
        for(int i = 0; i < Nx; ++i) {
            vNext[IDX(i,0)] = v[IDX(i,0)];
        }
    }
    
    if(topRank == MPI_PROC_NULL) {                  //assign top BC
        for(int i = 0; i < Nx; ++i) {
            vNext[IDX(i,Ny-1)] = v[IDX(i,Ny-1)];
        }
    }
    
    if(leftRank == MPI_PROC_NULL) {                 //assign left BC
        for(int j = 0; j < Ny; ++j) {
            vNext[IDX(0,j)] = v[IDX(0,j)];
        }
    }

    if(rightRank == MPI_PROC_NULL) {                //assign right BC
        for(int j = 0; j < Ny; ++j) {
            vNext[IDX(Nx-1,j)] = v[IDX(Nx-1,j)];
        }
    }

    //ensure all communication completed
    MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);   
}

void LidDrivenCavity::ComputeVelocity(double* u0, double* u1) {
    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 1: Transfer Data and Compute Interior Points---------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//    
    //to compute velocities, processes only need to know data to right and above, hence only need to send down and to left
    double dxi = 1/dx;
    double dyi = 1/dy;


    MPI_Isend(s, Nx, MPI_DOUBLE, bottomRank, 1, comm_col_grid,&requests[1]);            //tag = 1 -> streamfunction data sent down
    cblas_dcopy(Ny,s,Nx,tempLeft,1);                                                    //now extract left data
    MPI_Isend(tempLeft,Ny,MPI_DOUBLE,leftRank, 2, comm_row_grid,&requests[2]);          //tag = 2 -> streamfunction data sent left

    //compute interior points while waiting to send
    #pragma omp parallel for schedule(dynamic) 
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) * dyi;     //compute velocity in x direction at every grid point from streamfunction
                u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) * dxi;     //compute velocity in y direction at every grid point from streamfunction
            }
        }

    //use blocking receive as boundary data needed for next step
    MPI_Recv(sTopData,Nx,MPI_DOUBLE,topRank,1,comm_col_grid,MPI_STATUS_IGNORE);
    MPI_Recv(sRightData,Ny,MPI_DOUBLE,rightRank,2,comm_row_grid,MPI_STATUS_IGNORE);
    
    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 2: Compute Velocities on Corners of Local Domain-----------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//
    if((Nx == 1 )&& (Ny == 1)) {
        //if local domain is single cell not on boundary, then need access to data from two processes
        if(!boundaryDomain) {
            u0[0] = (sTopData[0] - s[0]) *dyi;
            u1[0] = - (sRightData[0] - s[0]) *dxi;
        }
        //if cell is on top rank, impose velocity of 1 for u0; for all other boundaries, do nothing as velocity should be zero for no slip
        else if(topRank == MPI_PROC_NULL) {
            u0[0] = U;
        }
    }
    //unless at global left/right where BC is imposed, if local domain is column vector, do:
    else if((Nx == 1) && (Ny != 1) && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {    
        //compute 'top' and 'bottom' corners, unless at top/bottom boundaries
        if(bottomRank != MPI_PROC_NULL) {
            u0[0] = (s[1] - s[0]) * dyi;
            u1[0] = - (sRightData[0] - s[0]) * dxi;
        }

        if(topRank != MPI_PROC_NULL) {
            u0[Ny-1] = (sTopData[0] - s[0]) * dyi;
            u1[Ny-1] = - (sRightData[0] - s[0]) * dxi;
        }
    }
    //unless at global top/bottom where BC is imposed, if local domain is row vector, do:
    else if((Nx != 1) && (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {            
        //compute 'left' and 'right' corners, unless at left/right boundaries
        if(leftRank != MPI_PROC_NULL) {
            u0[0] = (sTopData[0] - s[0]) * dyi;
            u1[0] = - (s[1] - s[0]) * dxi;
        }

        if(rightRank != MPI_PROC_NULL) {
            u0[Nx-1] = (sTopData[Nx-1] - s[Nx-1]) * dyi;
            u1[Nx-1] = - (sRightData[0] - s[Nx-1]) * dxi;
        }
    }
    //compute corners of general case
    else{
        //compute bottom left corner of domain, unless process is on left or bottom boundary, as already have BC there
        //similar logic for bottom right, top left and top right corners respectively
        if(!((bottomRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            u0[IDX(0,0)] = (s[IDX(0,1)] - s[IDX(0,0)]) * dyi;
            u1[IDX(0,0)] = - (s[IDX(1,0)] - s[IDX(0,0)]) * dxi;
        }

        if(!((bottomRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
            u0[IDX(Nx-1,0)] = (s[IDX(Nx-1,1)] - s[IDX(Nx-1,0)]) * dyi;
            u1[IDX(Nx-1,0)] = - (sRightData[0] - s[IDX(Nx-1,0)]) * dxi;
        }

        if(!((topRank == MPI_PROC_NULL) || (leftRank == MPI_PROC_NULL))) {
            u0[IDX(0,Ny-1)] = (sTopData[0] - s[IDX(0,Ny-1)]) * dyi;
            u1[IDX(0,Ny-1)] = - (s[IDX(1,Ny-1)] - s[IDX(0,Ny-1)]) * dxi;
        }

        if(!((topRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
            u0[IDX(Nx-1,Ny-1)] = (sTopData[Nx-1] - s[IDX(Nx-1,Ny-1)]) * dyi;
            u1[IDX(Nx-1,Ny-1)] = - (sRightData[Ny-1] - s[IDX(Nx-1,Ny-1)]) * dxi;
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------------//
    //--------------------------------------Step 2: Compute Velocities on Edges of Local Domain-------------------------------------------//
    //------------------------------------------------------------------------------------------------------------------------------------//    
    
    //if column vector domain, compute edge unless at left or right global boundary as BC already imposed along entire column
    if((Nx == 1) && (Ny > 1) && !((leftRank == MPI_PROC_NULL) || (rightRank == MPI_PROC_NULL))) {
        for(int j = 1; j < Ny - 1; ++j) {
            u0[j] = (s[j+1] - s[j]) * dyi;
            u1[j] = - (sRightData[j] - s[j]) * dxi;
        }
    }
    //if row vector domain, compute edge unless at top or bottom global boundary as BC already imposed along entire row
    else if((Nx != 1) && (Ny == 1) && !((topRank == MPI_PROC_NULL) || (bottomRank == MPI_PROC_NULL))) {
        for(int i = 1; i < Nx - 1; ++i) {
            u0[i] = (sTopData[i] - s[i]) * dyi;
            u1[i] = - (s[i+1] - s[i]) * dxi;
        }
    }
    //compute for general case (exclude single cell case)
    else if((Nx != 1) && (Ny != 1)){
        //only compute bottom row between corners if not at bottom of grid
        //same logic for all other points
        if(bottomRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                u0[IDX(i,0)] = (s[IDX(i,1)] - s[IDX(i,0)]) * dyi;
                u1[IDX(i,0)] = - (s[IDX(i+1,0)] - s[IDX(i,0)]) * dxi;
            }
        }
            
        if(topRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                u0[IDX(i,Ny-1)] = (sTopData[i] - s[IDX(i,Ny-1)]) * dyi;
                u1[IDX(i,Ny-1)] = - (s[IDX(i+1,Ny-1)] - s[IDX(i,Ny-1)]) * dxi;
            }
        }
            
        if(leftRank != MPI_PROC_NULL) {
        for(int j = 1; j < Ny - 1; ++j) {
                u0[IDX(0,j)] = (s[IDX(0,j+1)] - s[IDX(0,j)]) * dyi;
                u1[IDX(0,j)] = - (s[IDX(1,j)] - s[IDX(0,j)]) * dxi;
            }
        }
            
        //only compute right coluymn if not at right of grid
        if(rightRank != MPI_PROC_NULL) {
            for(int j = 1; j < Ny - 1; ++j) {
                u0[IDX(Nx-1,j)] =  (s[IDX(Nx-1,j+1)] - s[IDX(Nx-1,j)]) * dyi;
                u1[IDX(Nx-1,j)] = - (sRightData[j] - s[IDX(Nx-1,j)]) * dxi;
            }
        }
    }

    //now impose top BC, where x velocity is U at top surface for no slip
    if(topRank == MPI_PROC_NULL) {
        for (int i = 0; i < Nx; ++i) {
            u0[IDX(i,Ny-1)] = U;
        }
    }

    //make sure all communications finished before proceeding; only 1 and 2 requests are initalised, so start pointer at element 0
    MPI_Waitall(2,requests+1,MPI_STATUSES_IGNORE);
}

void LidDrivenCavity::CreateCartGrid(MPI_Comm &cartGrid,MPI_Comm &rowGrid, MPI_Comm &colGrid){
    
    int worldRank, size;    
    
    //return rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this-> size = size;                                                 //assign to member variable
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));                                          //round sqrt to nearest whole number
    
    if((p*p != size) || (size < 1)) {                                    //if not a square number, print error and terminate program
        if(worldRank == 0)                                              //print only on root rank
            cout << "Invalid process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        exit(-1);
    }

    /* Set up Cartesian topology to represent the 'grid' domain of the lid driven cavity problem
    Treat root process as bottom left of grid, with Cartesian coordinates (i,j)
    Increasing i goes to the right and increasing j goes up*/

    const int dims = 2;                                                                     //2 dimensions in grid
    int gridSize[dims] = {p,p};                                                             //p processes per dimension
    int periods[dims] = {0,0};                                                              //grid is not periodic
    int reorder = 1;                                                                        //reordering of grid allowed
    int keep[dims];                                                                         //denotes which dimension to keep when finding subgrids

    MPI_Cart_create(MPI_COMM_WORLD,dims,gridSize,periods,reorder, &cartGrid);         //create Cartesian topology grid
    
    //create row communnicator in subgrid so process can communicate with other processes on row   
    keep[0] = 0;        
    keep[1] = 1;                                                        //keep all processes with same j coordinate i.e. same row
    MPI_Cart_sub(cartGrid, keep, &rowGrid);
    
    //create column communnicator in subgrid so process can communicate with other processes on column
    keep[0] = 1;        
    keep[1] = 0;                                                        //keep all processes with same i coordinate i.e. same column
    MPI_Cart_sub(cartGrid, keep, &colGrid);
}

void LidDrivenCavity::SplitDomainMPI(MPI_Comm &grid, int globalNx, int globalNy, double globalLx, double globalLy, 
                                    int &localNx, int &localNy, double &localLx, double &localLy, int &xStart, int &yStart) {
    
    int rem,size,gridRank;
    int dims = 2;
    int coords[2];

    MPI_Comm_size(MPI_COMM_WORLD, &size);                       //return total number of MPI ranks, size denotes total number of processes P
    MPI_Comm_rank(grid, &gridRank);
    MPI_Cart_coords(grid, gridRank, dims, coords);              //use process rank in Cartesian grid to generate coordinates
    
    //assume that P = p^2 is already verified and find p, the number of processes along each domain dimension
    int p = round(sqrt(size));
    localNx = globalNx / p;                                     //minimum local size x and y domain for each process
    localNy = globalNy / p;

    //first assign for y dimension
    rem = globalNy % p;                                         //remainder denotes how many processes need to take an extra grid point in y direction (or row)

    if(coords[0] < rem) {                                       //add 1 extra row to first rem processes
        localNy++;
        yStart = localNy * coords[0];                           //index denoting how the starting row of the local domain maps onto the global domain
    }
    else {
        yStart = (localNy + 1) * rem + localNy * (coords[0] - rem);      //starting row accounts for previous processes with +1 rows and +0 rows
    }

     //same logic for x dimension (same as above, replacing "row" with "column" and "y" with "x")
    rem = globalNx % p;
        
    if(coords[1] < rem) {
        localNx++;
        xStart = localNx * coords[1];
    }
    else {
        xStart = (localNx + 1) * rem + localNx * (coords[1] - rem);
    }

    localLx = (double) globalLx * localNx / globalNx;           //compute local domain length by considering ratio of local domain size to global domain size
    localLy = (double) globalLy * localNy / globalNy;
}

