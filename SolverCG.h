#pragma once

/**
 * @class SolverCG
 * @brief Describes a preconditioned conjugate gradient solver that solves the equation \f$ -\nabla ^ 2 x = b \f$ 
 * Describes a preconditioned conjugate gradient solver which solves the matrix equation Ax=b, with max iteration number of 5000,
and error tolerance of 1e-3. In this context, A describes the coefficients of a second-order central-difference discretisation of the
operator \f$ -\nabla^2 \f$, x describes the streamfunction and b describes the vorticity. 
The problem domain is \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$, where \f$ L_x \f$ is the domain length in x direction and \f$ L_y \f$ is the 
domain length in the y direction.
 * @note When implemented with MPI, SolverCG expects inputs to already be discretised into local domains. All member variables describe
 the local problem domain, unless otherwise specified
 ******************************************************************************************************************************************/
class SolverCG
{
public:
    /**
     * @brief Constructor to create the solver by specifying the spatial domain of the problem \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$
     * @param[in] pNx   Number of grid points in x direction
     * @param[in] pNy   Number of grid points in y direction
     * @param[in] pdx   Grid spacing in x direction, should satisfy pdx = Lx/(pNx - 1) where Lx is domain length in x direction
     * @param[in] pdy   Grid spacing in y direction, should satisfy pdy = Ly/(pNy - 1) where Ly is domain length in y direction
     * @param[in] rowGrid   MPI communicator for the process row in Cartesian topology grid
     * @param[in] colGrid   MPI communicator for the process column in Cartesian topology grid
     ***************************************************************************************************************************************/
    SolverCG(int pNx, int pNy, double pdx, double pdy,MPI_Comm &rowGrid, MPI_Comm &colGrid);
    
    /**
     * @brief Destructor to deallocate memory
     ***************************************************************************************************************************************/ 
    ~SolverCG();

    /**
     * @defgroup GetSCG Get SolverCG Domain Parameters
     * Return a value that describes the local problem domain stored in a process, for testing purposes
     * @return The value that describes an aspect of the problem domain
     * @note Returned values describe the local domain of a process, which are not necessarily the same as the global domain (unless MPI ranks = 1)
     * @{
     ****************************************************************************************************************************************/
    double GetDx();             ///< Get the x step size parameter dx, for testing purporses
    double GetDy();             ///< Get the y step size parameter dy, for testing purporses
    int GetNx();                ///< Get the number of grid points in x direction, for testing purporses
    int GetNy();                ///< Get the number of grid points in y direction, for testing purporses
    /**@}*/

    /**
     * @brief Computes the solution to \f$ -\nabla ^ 2 x = b \f$ via a preconditioned conjugate gradient method. 
     * This equation is formulated as \f$ Ax=b \f$. Note that A describes the coefficients of a 
     * second-order central-difference discretisation of the operator \f$ -\nabla^2 \f$
     * @param[in] b     The desired result; in this context, the vorticity
     * @param x     On input, initial guess \f$ x_0 \f$; on output the computed solution (in this context, the streamfunction)
     */
    void Solve(double* b, double* x);

private:
    double dx;      ///<Grid spacing in x direction
    double dy;      ///<Grid spacing in y direction
    int Nx;         ///<Number of grid points in x direction
    int Ny;         ///<Number of grid points in y direction
    double* r;      ///<Variable for preconditioned conjugate gradient solver
    double* p;      ///<Variable for preconditioned conjugate gradient solver
    double* z;      ///<Variable for preconditioned conjugate gradient solver
    double* t;      ///<Variable for preconditioned conjugate gradient solver

    MPI_Comm comm_row_grid;                 ///<MPI communicator for the process row in Cartesian topology grid
    MPI_Comm comm_col_grid;                 ///<MPI communicator for the process column in Cartesian topology grid
    int size;                               ///<Size of a row/column communicator, where size*size is the total number of processors
    int globalNx;                           ///<Number of grid points in global domain in x direction
    int globalNy;                           ///<Number of grid points in global domain in y direction

    int rowRank;        ///<rank of current process in #comm_row_grid
    int colRank;        ///<rank of current process in #comm_col_grid
    int topRank;        ///<rank of process above current process in Cartesian grid, equals -2 (MPI_PROC_NULL) if nothing above
    int bottomRank;     ///<rank of process to bottom of current process in Cartesian grid, equals -2 (MPI_PROC_NULL) if nothing below
    int leftRank;       ///<rank of process to left of current process in Cartesian grid, equals -2 (MPI_PROC_NULL) if nothing to left
    int rightRank;      ///<rank of process to right of current process in Cartesian grid, equals -2 (MPI_PROC_NULL) if nothing to right

    /// MPI_Request handle to check data send -> [0] = send to top, [1] = send to bottom, [2] = send left, [3] = send right
    MPI_Request requests[4];                    

    bool boundaryDomain;                        ///<denotes whether the process is at the boundary of the Cartesian grid

    double* topData;                            ///<Store data from top process in Cartesian grid
    double* bottomData;                         ///<Store data from bototm process in Cartesian grid
    double* leftData;                           ///<Store data from left process in Cartesian grid
    double* rightData;                          ///<Store data from right process in Cartesian grid
    
    double* tempLeft;                           ///<Temporarily stores data for left hand side of current local grid, to be sent left
    double* tempRight;                          ///<Temporarily stores data for right hand side of current local grid, to be sent right

    /**
     * @brief Applies the second-order central-difference discretisation of operator \f$ -\nabla^2 \f$ such that \f$ -\nabla^2 p = t \f$
     * @param[in] p     Input data that the operator is applied to
     * @param[out] t     Result of the discretisation \f$ -\nabla^2 p \f$
     ****************************************************************************************************************************************/
    void ApplyOperator(double* p, double* t);
    
    /**
     * @brief Preconditions the problem formulation Ax=b to prevent ill-condition and improve convergence rate
     * @param[in] p     Input original un-preconditioned Ax
     * @param[out] t     Output preconditioned Ax
     *****************************************************************************************************************************************/
    void Precondition(double* p, double* t);
    
    /**
     * @brief Impose zero boundary conditions around the edge of the matrix \f$ AX \f$
     * @param p     On input, the matrix \f$ AX \f$ ; on output, the matrix \f$ AX \f$ with imposed boundary conditions
     *****************************************************************************************************************************************/
    void ImposeBC(double* p);

};

