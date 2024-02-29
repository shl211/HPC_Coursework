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
    const int Nx = 10;
    const int Ny = 10;
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
        x[i] = (double) rand()/RAND_MAX;
    }

    for(int i = 0; i < Nx; ++i){
        for(int j = 0; j < Ny; ++j) {
            x[IDX(i,j)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }

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

    //should be within the tolerance specified in solverCG
    for(int i = 0; i < n; i++) {
        BOOST_CHECK_SMALL(x[i]-x_actual[i],1e-3);
    }

    delete[] x;
    delete[] x_actual;
    delete[] b;
}