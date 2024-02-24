#pragma once

/**
 * @class SolverCG
 * @brief Describes an explicit forward time-integration serial solver to compute the spatial derivatives?
 */
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    void Solve(double* b, double* x);

private:
    double dx;      ///grid spacing in x direction
    double dy;      ///grid spacing in y direction
    int Nx;         ///number of grid points in x direction
    int Ny;         ///number of grid points in y direction
    double* r;      ///temp variables
    double* p;
    double* z;
    double* t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};

