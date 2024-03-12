#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

#include "SolverCG.h"

/**
 * @brief Macro to map matrix entry i,j onto it's corresponding location in memory, assuming column-wise matrix storage
 * @param I     matrix index i denoting the ith row
 * @param J     matrix index j denoting the jth columns
 */
#define IDX(I,J) ((J)*Nx + (I))                     //define a new operation to improve computation?

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy,MPI_Comm &rowGrid, MPI_Comm &colGrid)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    int n = Nx*Ny;                                  //total number of grid points
    r = new double[n];                              //allocate arrays size with total number of grid points
    p = new double[n];
    z = new double[n];
    t = new double[n];
    
    leftData = new double[Ny];
    rightData = new double[Ny];
    tempLeft = new double[Ny];
    tempRight = new double[Ny];
    topData = new double[Nx];
    bottomData = new double[Nx];            //data storage for data from other processes
    
    //MPI communication stuff
    comm_row_grid = rowGrid;
    comm_col_grid = colGrid;
    MPI_Comm_size(comm_row_grid,&size);     //get size of communicator

    //compute ranks along the row communciator and along teh column communicator
    MPI_Comm_rank(comm_row_grid, &rowRank); 
    MPI_Comm_rank(comm_col_grid, &colRank);

    //compute ranks for adjacent grids for data transfer, if at boundary, returns -2 (MPI_PROC_NULL)
    MPI_Cart_shift(comm_col_grid,0,1,&bottomRank,&topRank);//from bottom to top
    MPI_Cart_shift(comm_row_grid,0,1,&leftRank,&rightRank);//from left to rigth
    
    if((topRank != MPI_PROC_NULL) & (bottomRank != MPI_PROC_NULL) & (leftRank != MPI_PROC_NULL) & (rightRank != MPI_PROC_NULL))
        boundaryDomain = false;
    else
        boundaryDomain = true;      //check whether the current process is on the edge of the grid/cavity    
}

SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
    
    delete[] leftData;
    delete[] rightData;
    delete[] topData;
    delete[] bottomData;

    delete[] tempLeft;//buffers for sending data
    delete[] tempRight;
}

//getter functions for testing purposes
double SolverCG::GetDx() {
    return dx;
}

double SolverCG::GetDy() {
    return dy;
}

int SolverCG::GetNx() {
    return Nx;
}

int SolverCG::GetNy() {
    return Ny;
}

void SolverCG::Solve(double* b, double* x) {
    unsigned int n = Nx*Ny;                         //total number of grid points
    int k;                                          //counter to track iteration number
    double alphaNum;                                   //variables for conjugate gradient algorithm#
    double alphaDen;
    double betaNum;
    double betaDen;
    double eps;                                     //error                                     !! error or error squared !!
    double tol = 0.001;                             //error tolerance
    
    //global variables
    double globalAlpha;
    double globalAlphaTemp;
    double globalBeta;
    double globalBetaTemp;
    double globalEps;

    eps = cblas_dnrm2(n, b, 1);                     //if 2-norm of b is lower than tolerance squared, then b practically zero

    eps *= eps;//square it for summation
    //need to compute sum of error norms across all process for comparison, don't use local
    MPI_Allreduce(&eps,&globalEps,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    globalEps = sqrt(globalEps);//reduction gives norm squared

    if (globalEps < tol*tol) {                        
        std::fill(x, x+n, 0.0);                     //hence solution x is practically 0, output the 2-norm and exit function
        if((rowRank == 0) & (colRank == 0))
            cout << "Norm is " << globalEps << endl;          //don't waste time with algorithm, print on one rank
        return;
    }
    
    // --------------------------- PRECONDITIONED CONJUGATE GRADIENT ALGORITHM ---------------------------------------------------//
    //Refer to standard notation provided in the literature for this algorithm 
    ApplyOperator(x, t);                            //apply discretised operator -nabla^2 to x, so t = -nabla^2 x, or t = Ax 
    cblas_dcopy(n, b, 1, r, 1);                     // r_0 = b (i.e. b)
    ImposeBC(r);                                    //apply zeros to edges

    cblas_daxpy(n, -1.0, t, 1, r, 1);               //r=r-t (i.e. r = b - Ax) gives first step of conjugate gradient algorithm
    Precondition(r, z);                             //Precondition the problem, preconditioned matrix in z
    cblas_dcopy(n, z, 1, p, 1);                     // p_0 = z_0 (where z_0 is the preconditioned version of r_0)

    k = 0;                                          //initialise iteration counter
    
    do {
        k++;

        ApplyOperator(p, t);                       //compute -nabla^2 p and store in t (effectively A*p_k)

        alphaDen = cblas_ddot(n, t, 1, p, 1);          // alpha = p_k^T*A*p_k
        alphaNum = cblas_ddot(n, r, 1, z, 1);         // compute alpha_k = (r_k^T*r_k) / (p_k^T*A*p_k)
        betaDen  = cblas_ddot(n, r, 1, z, 1);          // z_k^T*r_k (for later in the algorithm)

        MPI_Allreduce(&alphaDen,&globalAlphaTemp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);//sum up local p_k^T*A*p_k, denosminatot which is a dot product
        MPI_Allreduce(&alphaNum,&globalAlpha, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);//sum up local numerator of alpha, dot product

        globalAlpha = globalAlpha/globalAlphaTemp;    //compute actual alpha

        //issue with dot product of transpose -> HOW???-> if treat as vectors, then fine     
        cblas_daxpy(n,  globalAlpha, p, 1, x, 1);         // x_{k+1} = x_k + alpha_k*p_k
        cblas_daxpy(n, -globalAlpha, t, 1, r, 1);         // r_{k+1} = r_k - alpha_k*A*p_k
    
        eps = cblas_dnrm2(n, r, 1);                 //norm r_{k+1} is error between algorithm and solution, check error tolerance
        eps *= eps;
        //need to compute sum of error norms across all process for comparison, don't use local         
        MPI_Allreduce(&eps,&globalEps,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        globalEps = sqrt(globalEps);

        if (globalEps < tol*tol) {
            break;                                  //stop algorithm if solution is within specified tolerance
        }
        
        Precondition(r, z);                         //precondition r_{k+1} and store in z_{k+1}

        betaNum = cblas_ddot(n, r, 1, z, 1);    //compute beta_k = (r_{k+1}^T*r_{k+1}) / (r_k^T*r_k)

        cblas_dcopy(n, z, 1, t, 1);                 //copy z_{k+1} into t, so t now holds preconditioned r_{k+1}
        
        //collect numerator and denominator of beta and then compute actaul global beta, do as late as possible to allow processes to catch up
        MPI_Allreduce(&betaDen,&globalBetaTemp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(&betaNum,&globalBeta,1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        
        globalBeta = globalBeta / globalBetaTemp;            

        cblas_daxpy(n, globalBeta, p, 1, t, 1);           //t = t + beta_k*p_k i.e. p_{k+1} = z_{k+1} + beta_k*p_k
        cblas_dcopy(n, t, 1, p, 1);                 //copy z_{k+1} from t into p, so p_{k+1} = z{k+1}, ready for next iteration
    } while (k < 5000);                             // max 5000 iterations

    if (k == 5000) {
        //only print on one process
        if((rowRank == 0) & (colRank == 0))
            cout << "FAILED TO CONVERGE" << endl;
        exit(-1);                                   //if 5000 iterations reached, no converged solution, so terminate program
    }                                               //otherwise, output results and details from the algorithm

    //only print on one process
    if((rowRank == 0) & (colRank == 0))
        cout << "Converged in " << k << " iterations. eps = " << globalEps << endl;

    }


void SolverCG::ApplyOperator(double* in, double* out) {
    //cout << "Rank " << rowRank << " has something on left " << leftRank  << endl;
    //first, send 'in' data that is required by adjacent grids
    //data stored rowwise -> see IDX definition

    //easy to send top and bottom data, so send first
    if(bottomRank != MPI_PROC_NULL) {           //only send to bottom if not on bottom boundary
        MPI_Isend(in,Nx,MPI_DOUBLE,bottomRank,6,comm_col_grid,&dataToDown);   //send data on bottom of current process down -> tag 6
    }
    
    if(topRank != MPI_PROC_NULL) {      //only send to top if not on top boundary
        MPI_Isend(in+Nx*(Ny-1), Nx, MPI_DOUBLE, topRank, 7, comm_col_grid,&dataToUp);    //send dataa on top of current process up -> tag 7
    }

    //now, first extract relevant daata for left and right columns
    cblas_dcopy(Ny,in,Nx,tempLeft,1);//write into temporary buffer to prevent accidental data overwrite with Isend
    cblas_dcopy(Ny,in+Nx-1,Nx,tempRight,1);

    //send left right data
    if(leftRank != MPI_PROC_NULL) { //only to send to left if not on left boundary
        MPI_Isend(tempLeft,Ny,MPI_DOUBLE,leftRank,4,comm_row_grid,&dataToLeft);   //send data on LHS of current process to the left -> tag 4
    }

    if(rightRank != MPI_PROC_NULL) { //only send to right if not on right boundary 
        MPI_Isend(tempRight,Ny, MPI_DOUBLE, rightRank, 5, comm_row_grid,&dataToRight);    //send data on RHS of current process to right -> tag 5
    }
    
    //compute interior points first to give time for all processes to send data; above data required for boundaries of each process
    //interior points of each local domain only requires knowledge of itself, nno other data required
    
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;                                    //optimised code, compute and store 1/(dx)^2 and 1/(dy)^2
    double dy2i = 1.0/dy/dy;
    int jm1 = 0, jp1 = 2;                                       //jm1 is j-1, jp1 is j+1; this allows for vectorisation of operation

    //only bother computing interior points if interior points actually exist
    if((Nx > 2) & (Ny > 2)) {
        for (int j = 1; j < Ny - 1; ++j) {                          //compute this only for interior grid points
            for (int i = 1; i < Nx - 1; ++i) {                      //i denotes x grids, j denotes y grids
                out[IDX(i,j)] = ( -     in[IDX(i-1, j)]             //note predefinition of IDX(i,j) = j*Nx + i, which yields location of entry i,j in memory
                                  + 2.0*in[IDX(i,   j)]
                                  -     in[IDX(i+1, j)])*dx2i       //calculates part of equation for second-order central-difference related to x
                              + ( -     in[IDX(i, jm1)]
                                  + 2.0*in[IDX(i,   j)]
                                  -     in[IDX(i, jp1)])*dy2i;      //calculates part of equation for second-order central-difference related to y
            }
            jm1++;                                                  //jm1 and jp1 incremented separately outside i loop
            jp1++;                                                  //this is done instead of j-1,j+1 in inner loop to encourage vectorisation
        }
    }
    
    //now receive the data, LHS RHS first as sent first
    if(leftRank != MPI_PROC_NULL) {     //can only receive from left rank if not on left boundary
        MPI_Recv(leftData,Ny,MPI_DOUBLE,leftRank,5,comm_row_grid,MPI_STATUS_IGNORE);            //rececive LHS data from the left process
    }
    
    if(rightRank != MPI_PROC_NULL) {
        MPI_Recv(rightData,Ny,MPI_DOUBLE,rightRank,4,comm_row_grid,MPI_STATUS_IGNORE);      //receive RHS data for current process from right process (which sent their left data)
    }
    
    if(topRank != MPI_PROC_NULL) {
        MPI_Recv(topData, Nx, MPI_DOUBLE, topRank, 6, comm_col_grid, MPI_STATUS_IGNORE); //receive data sent down from process above, is top row of current process
    }
    
    if(bottomRank != MPI_PROC_NULL) {
        MPI_Recv(bottomData, Nx, MPI_DOUBLE, bottomRank, 7, comm_col_grid,MPI_STATUS_IGNORE);//receive data sent up from process below, is bottom row of curren t process
    }
    
    //wait for send processes to complete, then free in input buffer
    if(leftRank != MPI_PROC_NULL) { //check whetehr data send to left is complete
        MPI_Wait(&dataToLeft,MPI_STATUS_IGNORE);
    }
    
    if(rightRank != MPI_PROC_NULL) { //check whetehr data send to right is complete
        MPI_Wait(&dataToRight,MPI_STATUS_IGNORE);
    }
    
    if(topRank != MPI_PROC_NULL) { //check whetehr data send up is complete
        MPI_Wait(&dataToUp,MPI_STATUS_IGNORE);
    }
    
    if(bottomRank != MPI_PROC_NULL) { //check whetehr data send down is complete
        MPI_Wait(&dataToDown,MPI_STATUS_IGNORE);
    }

    //first compute the 'corners' of each process domain
    if(Nx == 1 & Ny == 1 & !boundaryDomain) {//if single cell not on boundary, then need access to data from four processes
        out[0] = ( - leftData[0] + 2.0*in[0] - rightData[0] ) * dx2i
                + (- bottomData[0] + 2*in[0] - topData[0] ) * dy2i; 
    }
    else if((Nx == 1) & (Ny != 1) & !((leftRank == MPI_PROC_NULL) | (rightRank == MPI_PROC_NULL)) ) {
        //if process effectively a colmumn vector, do this, unless it's at left or right boundary
        //compute 'top' corner, unless at top grid boundary already, need access to three pieces of data
        if(topRank != MPI_PROC_NULL) {
            out[Ny-1] = (- leftData[Ny-1] + 2.0*in[Ny-1] - rightData[Ny-1] ) * dx2i
                    + (- in[Ny-2] + 2*in[Ny-1] - topData[0]) * dy2i; 
        }
        
        //compute 'bottom' corner, unless at bottom grid boundary already
        if(bottomRank != MPI_PROC_NULL) {
            out[0] = (- leftData[0] + 2.0 * in[0] - rightData[0] ) * dx2i
                    + (- bottomData[0] + 2*in[0] - in[1]) * dy2i;
        }
    }
    else if((Nx != 1) & (Ny == 1) & !((topRank == MPI_PROC_NULL) | (bottomRank == MPI_PROC_NULL)) ) {
        //if process effectively a row domain, do this, unless already at top or bottom boundaries
        //compute 'left' corner, unless at LHS of grid already
        if(leftRank != MPI_PROC_NULL) {
            out[0] = ( - leftData[0] + 2.0 * in[0] - in[1] ) * dx2i
                    + (- bottomData[0] + 2*in[0] - topData[0]) * dy2i;
        }
        
        //compute 'right' corner, unless at RHS of grid already
        if(rightRank != MPI_PROC_NULL) {
            out[Nx-1] = (- in[Nx-2] + 2.0 * in[0] - rightData[1] ) * dx2i
                    +(- bottomData[Nx-1] + 2 * in[0] - topData[Nx-1] ) * dy2i;
        }
    }
    else {  //otherwise, for general case, compute the four corners
        
        //compute bottom left corner of domain, unless process is on left or bottom boundary, as already have BC there
       if(!((bottomRank == MPI_PROC_NULL) | (leftRank == MPI_PROC_NULL))) {
            out[IDX(0,0)] = (- leftData[0] + 2.0*in[IDX(0,0)] - in[IDX(1,0)]) * dx2i
                        + (- bottomData[0] + 2.0*in[IDX(0,0)] - in[IDX(0,1)]) * dy2i;
       }

        //compute bottom right corner of domain, unless process is on right or bottom boundary
        if(!((bottomRank == MPI_PROC_NULL) | (rightRank == MPI_PROC_NULL))) {
            out[IDX(Nx-1,0)] = (- in[IDX(Nx-2,0)] + 2.0*in[IDX(Nx-1,0)] - rightData[0]) * dx2i
                        + (- bottomData[Nx-1] + 2.0*in[IDX(Nx-1,0)] - in[IDX(Nx-1,1)]) * dy2i;
        }

        //compute top left corner of domain, unless process is on left or top boundary
        if(!((topRank == MPI_PROC_NULL) | (leftRank == MPI_PROC_NULL))) {
            out[IDX(0,Ny-1)] = (- leftData[Ny-1] + 2.0*in[IDX(0,Ny-1)] - in[IDX(1,Ny-1)]) * dx2i
                        + (- in[IDX(0,Ny-2)] + 2.0*in[IDX(0,Ny-1)] - topData[0]) * dy2i;

        }

        //compute top right corner of domain, unless process is on right or top boundary
        if(!((topRank == MPI_PROC_NULL) | (rightRank == MPI_PROC_NULL))) {
            out[IDX(Nx-1,Ny-1)] = (- in[IDX(Nx-2,Ny-1)] + 2.0*in[IDX(Nx-1,Ny-1)] - rightData[Ny-1]) * dx2i
                        + (- in[IDX(Nx-1,Ny-2)] + 2.0*in[IDX(Nx-1,Ny-1)] - topData[Nx-1]) * dy2i;
        
        }
    }
        
    //now compute process edges
    if((Nx == 1) & (Ny > 1) & !((leftRank == MPI_PROC_NULL) | (rightRank == MPI_PROC_NULL))) {
        //if column vector, don't need to do for left or right as BC already imposed along entire column
        for(int j = 1; j < Ny - 1; ++j) {
            out[j] = ( - leftData[j] + 2.0 * in[j] - rightData[j]) * dx2i
                    + ( -  in[j-1] + 2.0 * in[j] - in[j+1]) * dy2i;
        }
    }
    else if((Nx != 1) & (Ny == 1) & !((topRank == MPI_PROC_NULL) | (bottomRank == MPI_PROC_NULL))) {
        //if row vector, don't need to do for top and bottom rows as BC already imposed along entire row
        for(int i = 1; i < Nx - 1; ++i) {
            out[i] = ( - in[i-1] + 2.0 * in[i] - in[i+1] ) * dx2i
                    + ( - bottomData[i] + 2.0 * in[i] - topData[i]) * dy2i;
        }
    }
    else {  //otherwise, for teh general case, compute process edge data
        //only compute bottom row if not at bottom of grid
        if(bottomRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                out[IDX(i,0)] = (- in[IDX(i-1,0)] + 2.0*in[IDX(i,0)] - in[IDX(i+1,0)] ) * dx2i
                            + ( - bottomData[i] + 2.0*in[IDX(i,0)] - in[IDX(i,1)] ) * dy2i;
            }
        }
        
        //only compute top row if not at top of grid
        if(topRank != MPI_PROC_NULL) {
            for(int i = 1; i < Nx - 1; ++i) {
                out[IDX(i,Ny-1)] = (- in[IDX(i-1,Ny-1)] + 2.0*in[IDX(i,Ny-1)] - in[IDX(i+1,Ny-1)] ) * dx2i
                            + ( - in[IDX(i,Ny-2)] + 2.0 * in[IDX(i,Ny-1)] - topData[i]) * dy2i;
            }
        }
        
        //only compute left column if not at left of grid
        if(leftRank != MPI_PROC_NULL) {
            for(int j = 1; j < Ny - 1; ++j) {
                out[IDX(0,j)] = (- leftData[j] + 2.0*in[IDX(0,j)] - in[IDX(1,j)] ) * dx2i
                            + ( - in[IDX(0,j-1)] + 2.0*in[IDX(0,j)] - in[IDX(0,j+1)] ) * dy2i;
            }
        }
        
        //only compute right coluymn if not at right of grid
        if(rightRank != MPI_PROC_NULL) {
            for(int j = 1; j < Ny - 1; ++j) {
                out[IDX(Nx-1,j)] = (- in[IDX(Nx-2,j)] + 2.0*in[IDX(Nx-1,j)] - rightData[j] ) * dx2i
                            + ( - in[IDX(Nx-1,j-1)] + 2.0*in[IDX(Nx-1,j)] - in[IDX(Nx-1,j+1)] ) * dy2i;
            }

        }
    }
}

void SolverCG::Precondition(double* in, double* out) {
    int i, j;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;                                //optimised code, compute and store 1/(dx)^2 and 1/(dy)^2
    double factor = 2.0*(dx2i + dy2i);                      //precondition will involve dividing all terms by 2(1/dx/dx + 1/dy/dy)
    
    //first do process corners
    if( (leftRank == MPI_PROC_NULL) | (bottomRank == MPI_PROC_NULL)) {//if process is on the left or bottom, impose BC on bottom left corner 
        out[0] = in[0];
    }
    else {//otherwise, precondition
        out[0] = in[0]/factor;
    }
    
    if( (leftRank == MPI_PROC_NULL) | (topRank == MPI_PROC_NULL)) { //if process is on left ro top, impose BC on top left corner
        out[IDX(0,Ny-1)] = in[IDX(0,Ny-1)];
    }
    else{
        out[IDX(0,Ny-1)] = in[IDX(0,Ny-1)]/factor;
    }
    
    if( (rightRank == MPI_PROC_NULL) | (bottomRank == MPI_PROC_NULL)) {//if process is on right or bottom, impose BC on bottom right corner
        out[IDX(Nx-1,0)] = in[IDX(Nx-1,0)];
    }
    else{
        out[IDX(Nx-1,0)] = in[IDX(Nx-1,0)]/factor;        
    }    
    
    if((rightRank == MPI_PROC_NULL) | (topRank == MPI_PROC_NULL)) {//if process is on right or top, impose BC on top right corner
        out[IDX(Nx-1,Ny-1)] = in[IDX(Nx-1,Ny-1)];
    }
    else{
        out[IDX(Nx-1,Ny-1)] = in[IDX(Nx-1,Ny-1)]/factor;
    }
    
    //now compute for process edges
    if( leftRank != MPI_PROC_NULL) {        //if process not on left boundary, precodnition LHS, otherwise maintain same BC
        for(j = 1; j < Ny-1; ++j) {
            out[IDX(0,j)] = in[IDX(0,j)]/factor;
        }
    }
    else {
        for(j = 1; j < Ny-1; ++j) {
            out[IDX(0,j)] = in[IDX(0,j)];
        }
    }
    
    if(rightRank != MPI_PROC_NULL) {//if process not on right boundary, precodnition RHS, otherwise maintain same BC
        for(j = 1; j < Ny - 1; ++j) {
            out[IDX(Nx-1,j)] = in[IDX(Nx-1,j)]/factor;
        }
    }
    else {
        for(j = 1; j < Ny - 1; ++j) {
            out[IDX(Nx-1,j)] = in[IDX(Nx-1,j)];
        }
    }
    
    if(bottomRank != MPI_PROC_NULL) {   //if process not on bottom boundary, precodntion bottom row, otherwise maintain same BC
        for(i = 1; i < Nx - 1; ++i) {
            out[IDX(i,0)] = in[IDX(i,0)]/factor;
        }
    }
    else {
         for(i = 1; i < Nx - 1; ++i) {
            out[IDX(i,0)] = in[IDX(i,0)];
        }
    }
    
    if(topRank != MPI_PROC_NULL) {   //if process not on top boundary, precodntion top row, otherwise maintain same BC
        for(i = 1; i < Nx - 1; ++i) {
            out[IDX(i,Ny-1)] = in[IDX(i,Ny-1)]/factor;
        }
    }
    else {
         for(i = 1; i < Nx - 1; ++i) {
            out[IDX(i,Ny-1)] = in[IDX(i,Ny-1)];
        }
    }
    
    //precondition all other interior points
    for (i = 1; i < Nx - 1; ++i) {
        for (j = 1; j < Ny - 1; ++j) {                  
            out[IDX(i,j)] = in[IDX(i,j)]/factor;            //apply precondition, reduce condition number
        }
    }
}

void SolverCG::ImposeBC(double* inout) {
        
    //only impose BC on relevant boundaries of the boundary processes
    if(bottomRank == MPI_PROC_NULL) {       //if bottom process, impose BC on bottom row
        for(int i = 0; i < Nx; ++i) {
            inout[IDX(i,0)] = 0.0;
        }
    }
    
    if(topRank == MPI_PROC_NULL) {
        for(int i = 0; i < Nx; ++i) {
            inout[IDX(i,Ny-1)] = 0.0;       //BC on top row
        }
    }
    
    if(leftRank == MPI_PROC_NULL) {
        for(int j = 0; j < Ny; ++j) {
            inout[IDX(0,j)] = 0.0;          //BC on left column
        }
    }
    
    if(rightRank == MPI_PROC_NULL) {
        for(int j = 0; j < Ny; ++j) {
            inout[IDX(Nx-1,j)] = 0.0;       //BC on right column
        }
    }
}