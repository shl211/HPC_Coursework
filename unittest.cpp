#include "SolverCG.h"

#define BOOST_TEST_MODULE SolverCG
#include <boost/test/included/unit_test.hpp>


//test solver, implicity tests constructor and all other private functions
BOOST_AUTO_TEST_CASE(Solver)
{
    int Nx = 10;
    int Ny = 10;
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
    

}