#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

#include <cblas.h>

#include "SolverCG.h"

#define IDX(I,J) ((J)*Nx + (I)) //define a new operation

/** 
 * @brief Constructor to create the solver by specifying the problem spatial domain \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$
 * @param pNx   Number of grid points in x direction
 * @param pNy   Number of grid points in y direction
 * @param pdx   Grid spacing in x direction, should satisfy pdx = Lx/(pNx - 1) where Lx is domain length in x direction
 * @param pdy   Grid spacing in y direction, should satisfy pdy = Ly/(pNy - 1) where Ly is domain length in y direction
 */
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    int n = Nx*Ny;                  //total number of grid points
    r = new double[n];              //allocate temp arrays size with total number of grid points
    p = new double[n];
    z = new double[n];
    t = new double[n];              //temp
}


/**
 * @brief Destructor to deallocate memory called on heap by solver
 */
SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
}

/**
 * @brief Executes conjugate gradient algorithm to solve the spatial problem \f$ Ax=b \f$, where A contains second order 
 * central difference coefficients and boundary conditions, x is streamfunctions, and b is vorticity
 * @param b     Pointer to array containing the vorticity at each grid point
 * @param x     Pointer to array containing the streamfunctions at each grid point
 */
void SolverCG::Solve(double* b, double* x) {
    unsigned int n = Nx*Ny;                     //total number of grid points
    int k;
    double alpha;                               //variables for conjugate gradient algorithm
    double beta;
    double eps;//error
    double tol = 0.001;                             

    eps = cblas_dnrm2(n, b, 1);                 //if 2-norm of b is lower than tolerance squared, then b practically zero
    if (eps < tol*tol) {                        
        std::fill(x, x+n, 0.0);                 //hence solution x is practically 0, output the 2-norm and exit function
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t);                        //discretise nabla psi with second order central difference, store coefficients in t (effectively Ax)
    cblas_dcopy(n, b, 1, r, 1);                 // r_0 = b (i.e. b), so r denotes vorticity
    ImposeBC(r);                                //fluid at rest, so apply initial vorticity BC to r

    cblas_daxpy(n, -1.0, t, 1, r, 1);           //r=r-t (i.e. r = b - Ax) gives first step of conjugate gradient algorithm
    Precondition(r, z);                         //Precondition the problem, preconditioned matrix in z
    cblas_dcopy(n, z, 1, p, 1);                 // p_0 = r_0 (where r_0 is the preconditioned matrix z)

    k = 0;//initialise counter
    do {
        k++;
        // Perform action of Nabla^2 * p
        ApplyOperator(p, t);                    //compute nabla^2 p and store in t (effectively A*p_k)

        alpha = cblas_ddot(n, t, 1, p, 1);      // alpha = p_k^T A p_k
        alpha = cblas_ddot(n, r, 1, z, 1) / alpha; // compute alpha_k
        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k (for later in the algorithm)

        cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps = cblas_dnrm2(n, r, 1);         //norm r_{k+1} is error between algorithm and solution, check error tolerance

        if (eps < tol*tol) {
            break;
        }
        Precondition(r, z);                         //precondition r_{k+1} and store in z
        beta = cblas_ddot(n, r, 1, z, 1) / beta;    //compute beta_k

        cblas_dcopy(n, z, 1, t, 1);                 //copy z into t, so t now holds preconditioned r
        cblas_daxpy(n, beta, p, 1, t, 1);           //t = t + beta p i.e. p_{k+1} = r_{k+1} + beta_k p_k
        cblas_dcopy(n, t, 1, p, 1);                 //copy p_{k+1} from t into p

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    cout << "Converged in " << k << " iterations. eps = " << eps << endl;
}

/**
 * @brief Computes discretisation of nabla^2 \f$ \nabla^2 \f$ ( \f$ \psi \f$ or \f$ \omega \f$ ) via equation: 
 \f$ \omega_{i,j}^n = -(\frac{\psi_{i+1,j}^n-2\psi_{i,j}^n+\psi_{i,j+1}^n}{(\Deltax)^2}
 +\frac{\psi_{i+1,j}^n-2\psi_{i,j}^n+\psi_{i,j+1}^n}{(\Deltay)^2}) \f$ for interior grid points
 * @param in    pointer to matrix containing psi or omega at time t
 * @param out   pointer to matrix containing nabla^2psi or nabla^2 omega at time t
 */
void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;                                    //optimised code, compute and store 1/(dx)^2 and 1/(dy)^2
    double dy2i = 1.0/dy/dy;
    int jm1 = 0, jp1 = 2;                                       //jm1 is j-1, jp1 is j+1; this allows for vectorisation of operation
    for (int j = 1; j < Ny - 1; ++j) {                          //compute this only for interior grid points
        for (int i = 1; i < Nx - 1; ++i) {                      //i denotes x grids, j denotes y grids
            out[IDX(i,j)] = ( -     in[IDX(i-1, j)]             //note predefinition of IDX(i,j) = i*Nx + j, which yields array index
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i       //calculates first part of equation
                          + ( -     in[IDX(i, jm1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, jp1)])*dy2i;      //calculates second part of equation
        }
        jm1++;                                                  //jm1 and jp1 incremented separately outside i loop
        jp1++;
    }
}

/**
 * @brief Precondition matrix to reduce condition number and aid convergence of solution. Problem is now preceonditioned problem.
 * @param in    Pointer to array containing original matrix
 * @param out   Pointer to array containing preeconditioned matrix
 */
void SolverCG::Precondition(double* in, double* out) {
    int i, j;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 2.0*(dx2i + dy2i);
    for (i = 1; i < Nx - 1; ++i) {
        for (j = 1; j < Ny - 1; ++j) {                  
            out[IDX(i,j)] = in[IDX(i,j)]/factor;            //apply precondition, reduce condition number
        }
    }
    // Boundaries
    for (i = 0; i < Nx; ++i) {                              //maintain same boundary conditions
        out[IDX(i, 0)] = in[IDX(i,0)];                      //bottom
        out[IDX(i, Ny-1)] = in[IDX(i, Ny-1)];               //top
    }

    for (j = 0; j < Ny; ++j) {
        out[IDX(0, j)] = in[IDX(0, j)];                     //left
        out[IDX(Nx - 1, j)] = in[IDX(Nx - 1, j)];           //right
    }
}


/**
 * @brief Assign initial boundary conditions (zero to each side)
 * @param inout     Pointer to array containing streamfunctions at timestep t 
 */
void SolverCG::ImposeBC(double* inout) {
        // Boundaries
    for (int i = 0; i < Nx; ++i) {
        inout[IDX(i, 0)] = 0.0;             //zero BC on bottom surface
        inout[IDX(i, Ny-1)] = 0.0;          //zero BC on top surface
    }

    for (int j = 0; j < Ny; ++j) {
        inout[IDX(0, j)] = 0.0;             //zero BC on left surface
        inout[IDX(Nx - 1, j)] = 0.0;        //zero BC on right surface
    }
}