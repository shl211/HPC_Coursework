#pragma once

/**
 * @class SolverCG
 * @brief Describes a preconditioned conjugate gradient solver which solves the matrix equation Ax=b, with max iteration number of 5000,
and error tolerance of 1e-3. In this context, A describes the coefficients of a second-order central-difference discretisation of the
operator \f$ -\nabla^2 \f$, x describes the streamfunction and b describes the vorticity. The problem domain is  
 \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$, where \f$ L_x \f$ is the domain length in x direction and \f$ L_y \f$ is the domain length in the
 y direction.
 */
class SolverCG
{
public:
    /**
     * @brief Constructor to create the solver by specifying the spatial domain of the problem \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$
     * @param pNx   Number of grid points in x direction
     * @param pNy   Number of grid points in y direction
     * @param pdx   Grid spacing in x direction, should satisfy pdx = Lx/(pNx - 1) where Lx is domain length in x direction
     * @param pdy   Grid spacing in y direction, should satisfy pdy = Ly/(pNy - 1) where Ly is domain length in y direction
     */
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    
    /**
     * @brief Destructor
     */ 
    ~SolverCG();

    /**
     * @brief Computes the solution to Ax=b via a preconditioned conjugate gradient method. Note that A describes
     the coefficients of a second-order central-difference discretisation of the operator \f$ -\nabla^2 \f$
     * @param b     The desired result; in this context, the vorticity
     * @param x     The computed solution; in this context, the streamfunction
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

    /**
     * @brief Applies the second-order central-difference discretisation of operator \f$ -\nabla^2 \f$ such that \f$ -\nabla^2 p = t \f$
     * @param p     Input data that the operator is applied to
     * @param t     Result of the discretisation \f$ -\nabla^2 p \f$
     */
    void ApplyOperator(double* p, double* t);
    
    /**
     * @brief Preconditions the problem formulation Ax=b to prevent ill-condition and improve convergence rate
     * @param p     Input original un-preconditioned Ax
     * @param t     Output preconditioned Ax
     */
    void Precondition(double* p, double* t);
    
    /**
     * @brief Impose zero boundary conditions around the edge of the matrix AX
     * @param p     On input, the matrix AX; on output, the matrix AX with imposed boundary conditions
     */
    void ImposeBC(double* p);

};

