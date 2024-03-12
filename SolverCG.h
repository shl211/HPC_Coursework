#pragma once

/**
 * @class SolverCG
 * @brief Describes a preconditioned conjugate gradient solver which solves the matrix equation Ax=b, with max iteration number of 5000,
and error tolerance of 1e-3. In this context, A describes the coefficients of a second-order central-difference discretisation of the
operator \f$ -\nabla^2 \f$, x describes the streamfunction and b describes the vorticity. 
The problem domain is \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$, where \f$ L_x \f$ is the domain length in x direction and \f$ L_y \f$ is the 
domain length in the y direction.
 */
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
     */
    SolverCG(int pNx, int pNy, double pdx, double pdy,MPI_Comm &rowGrid, MPI_Comm &colGrid);
    
    /**
     * @brief Destructor to deallocate memory
     */ 
    ~SolverCG();

    /**
     * @brief Get the x step size parameter dx, for testing purporses
     * @return The x step size parameter dx
     */
    double GetDx();
    
    /**
     * @brief Get the y step size parameter dy, for testing purporses
     * @return The y step size parameter dy
     */
    double GetDy();
    
    /**
     * @brief Get the number of grid points in x direction Nx, for testing purporses
     * @return The number of grid points in x direction Nx
     */
    int GetNx();
     
   /**
     * @brief Get the number of grid points in y direction Ny, for testing purporses
     * @return The number of grid points in y direction Ny
     */
    int GetNy();

    /**
     * @brief Computes the solution to Ax=b via a preconditioned conjugate gradient method. Note that A describes
     the coefficients of a second-order central-difference discretisation of the operator \f$ -\nabla^2 \f$
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
    int globalNx;                               ///<global Nx
    int globalNy;                               ///<global Ny

    int rowRank;///<rank of current process in comm_row_grid
    int colRank;///<rank of current process in comm_col_grid
    int topRank;///<rank of process above current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing above
    int leftRank;///<rank of process to left of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to left
    int rightRank;///<rank of process to right of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to right
    int bottomRank;///<rank of process to bottom of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing below

    bool boundaryDomain; ///<denotes whether the process is at the boundary of the Cartesian grid

    /**
     * @brief Applies the second-order central-difference discretisation of operator \f$ -\nabla^2 \f$ such that \f$ -\nabla^2 p = t \f$
     * @param[in] p     Input data that the operator is applied to
     * @param[out] t     Result of the discretisation \f$ -\nabla^2 p \f$
     */
    void ApplyOperator(double* p, double* t);
    
    /**
     * @brief Preconditions the problem formulation Ax=b to prevent ill-condition and improve convergence rate
     * @param[in] p     Input original un-preconditioned Ax
     * @param[out] t     Output preconditioned Ax
     */
    void Precondition(double* p, double* t);
    
    /**
     * @brief Impose zero boundary conditions around the edge of the matrix AX
     * @param p     On input, the matrix AX; on output, the matrix AX with imposed boundary conditions
     */
    void ImposeBC(double* p);

};

