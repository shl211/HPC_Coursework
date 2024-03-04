#include <cmath>
#include <iostream>
#include <cstdlib>
#include "SolverCG.h"

#define BOOST_TEST_MODULE SolverCG
#include <boost/test/included/unit_test.hpp>

#define IDX(I,J) ((J)*Nx + (I)) //define a new operation

//test solver, implicity tests constructor and all other private functions
BOOST_AUTO_TEST_CASE(SolverCase1)
{
    const int Nx = 10;
    const int Ny = 10;
    double dx = 0.1;
    double dy = 0.1;    
    int n = Nx*Ny;
    
    double *b = new double[n];
    double *x = new double[n];
    
    SolverCG test(Nx,Ny,dx,dy);

    //first test out if b very small and close to 0
    for(int i = 0; i < n; i++) {
        b[i] = 1e-8;
    }
    
    test.Solve(b,x);
    
    for(int i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(x[i],0.0);
    }
    
    delete[] b;
    delete[] x;
}


BOOST_AUTO_TEST_CASE(SolverCase2) 
{
    const int k = 3;
    const int l = 3;
    const double Lx = 2.0 / k;//correct domain for problem, such that sin sin fits boundary conditions
    const double Ly = 2.0 / l;
    const int Nx = 100;
    const int Ny = 100;
    double dx = (double)Lx/(Nx - 1);
    double dy = (double)Ly/(Ny - 1);    
    int n = Nx*Ny;
    double *b = new double[n];
    double *x = new double[n];
    
    SolverCG test(Nx,Ny,dx,dy);
    
    //now test with sinusoidal analytical solution    
    //compute actual b based off analytical solution
    
    //initialise arrays
    std::srand(time(0));
    for(int i = 0; i < n; i++) {
        b[i] = 0.0;
        //x[i] = (double) rand()/RAND_MAX;
        x[i] = 0.1;
    }
    
    for (int i = 0; i < Nx; ++i) {
        x[IDX(i, 0)] = 0.0;             //zero BC on bottom surface
        x[IDX(i, Ny-1)] = 0.0;          //zero BC on top surface
    }

    for (int j = 0; j < Ny; ++j) {
        x[IDX(0, j)] = 0.0;             //zero BC on left surface
        x[IDX(Nx - 1, j)] = 0.0;        //zero BC on right surface
    }

    /*for(int i = 0; i < Nx; ++i){
        for(int j = 0; j < Ny; ++j) {
            x[IDX(i,j)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }*/

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            b[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    
    test.Solve(b,x);
    
    //compute actual x with analytical solution
    double* x_actual = new double[n];
    for(int i = 0; i < Nx; ++i){
        for(int j = 0; j < Ny; ++j) {
            x_actual[IDX(i,j)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }
    /*
    //-nabla^2 x
    double* b_check = new double[n];
    
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;                                    //optimised code, compute and store 1/(dx)^2 and 1/(dy)^2
    double dy2i = 1.0/dy/dy;
    int jm1 = 0, jp1 = 2;                                       //jm1 is j-1, jp1 is j+1; this allows for vectorisation of operation
    for (int j = 1; j < Ny - 1; ++j) {                          //compute this only for interior grid points
        for (int i = 1; i < Nx - 1; ++i) {                      //i denotes x grids, j denotes y grids
            b_check[IDX(i,j)] = ( -     x[IDX(i-1, j)]             //note predefinition of IDX(i,j) = i*Nx + j, which yields array index
                              + 2.0*x[IDX(i,   j)]
                              -     x[IDX(i+1, j)])*dx2i       //calculates first part of equation
                          + ( -     x[IDX(i, jm1)]
                              + 2.0*x[IDX(i,   j)]
                              -     x[IDX(i, jp1)])*dy2i;      //calculates second part of equation
        }
        jm1++;                                                  //jm1 and jp1 incremented separately outside i loop
        jp1++;
    }
    
    for(int i = 0; i < n; i++) {
        std::cout << "b_check " << b_check[i] << " b_actual " << b[i] << std::endl;
    }
    delete[] b_check;*/

    /*for(int i = 0; i < n; i++) {
        std::cout << "CG X " << x[i] << " Actual" << x_actual[i] << std::endl;
    }*/

    //should be within the tolerance specified in solverCG
    for(int i = 0; i < n; i++) {
        //BOOST_REQUIRE(abs(x[i] - x_actual[i]) < 1e-3);
        //std::cout << x[i] << std::endl;
        BOOST_CHECK_SMALL(x[i]-x_actual[i],1e-3);
    }

    delete[] x;
    delete[] x_actual;
    delete[] b;
}

