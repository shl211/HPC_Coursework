/**
 * @brief Boost unit tests of classes SolverCG and LidDrivenCavity
 */
#define BOOST_TEST_MODULE main
#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <streambuf>
#include <cmath>
#include <cstdio>
#include <cblas.h>
#include <mpi.h>

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Macro to map matrix entry i,j onto it's corresponding location in memory, assuming column-wise matrix storage
 * @param I     matrix index i denoting the ith row
 * @param J     matrix index j denoting the jth columns
 */
#define IDX(I,J) ((J)*localNx + (I))                     //define a new operation to improve computation?

struct MPISetUp {
    MPISetUp() {
        // Access argc and argv from Boost Test framework
        int& argc = boost::unit_test::framework::master_test_suite().argc;
        char**& argv = boost::unit_test::framework::master_test_suite().argv;

        MPI_Init(&argc, &argv);
    }

    ~MPISetUp() {
        MPI_Finalize();
    }
};

BOOST_GLOBAL_FIXTURE(MPISetUp);

/**
 * @brief Setup Cartesian grid and column and row communicators
 * @param[out] comm_Cart_Grid   Communicator for Cartesian grid
 * @param[out] comm_row_grid    Communicator for current row of Cartesian grid
 * @param[out] comm_col_grid    Communicator for current column of Cartesian grid
 * @param[out] size     size of communicators
 */
void CreateCartGridVerify(MPI_Comm &comm_Cart_Grid,MPI_Comm &comm_row_grid, MPI_Comm &comm_col_grid){
    
    int worldRank, size;    
    
    //return rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));   //round sqrt to nearest whole number
    
    if((p*p != size) | (size < 1)) {                   //if not a square number, print error and terminate program
        if(worldRank == 0)                                       //print only on root rank
            cout << "Invalide process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        exit(-1);
    }

    //set up Cartesian topology to represent the 'grid' nature of the problem
    const int dims = 2;                                 //2 dimensions in grid
    int gridSize[dims] = {p,p};                         //p processes per dimension
    int periods[dims] = {0,0};                          //grid is not periodic
    int reorder = 1;                                    //reordering of grid
    MPI_Cart_create(MPI_COMM_WORLD,dims,gridSize,periods,reorder, &comm_Cart_Grid);        //create Cartesian topology
    
    //extract coordinates
    int gridRank;
    int coords[dims];
    int keep[dims];
    
    MPI_Comm_rank(comm_Cart_Grid, &gridRank);         //retrieve rank in grid, also check if grid created successfully
    MPI_Cart_coords(comm_Cart_Grid, gridRank, dims, coords);        //generate coordinates
    
    keep[0] = 0;        //create row communnicator in subgrid, process can communicate with other processes on row
    keep[1] = 1;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_row_grid);
    
    keep[0] = 1;        //create column communnicator in subgrid, process can communicate with other processes on column
    keep[1] = 0;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_col_grid);
}

/**
 * @brief Split the global grid size into local grid size based off MPI grid size
 * @param[in] grid      MPI Cartesian grid
 * @param[in] globalNx  Global Nx domain to be discretised
 * @param[in] globalNy Global Ny domain to be discretised
 * @param[out] localNx  Domain size Nx for each local process
 * @param[out] localNy  Domain size Ny for each local process
 * @param[out] xStart   Starting point of local domain in global domain, x direction
 * @param[out] yStart   Starting point of local domain in global domain, y direction
 */
void SplitDomainMPIVerify(MPI_Comm &grid, int globalNx, int globalNy, double globalLx, double globalLy, 
                        int &localNx, int &localNy, double &localLx, double &localLy, int &xStart, int &yStart) {
    
    int xDomainSize,yDomainSize;                        //local domain sizes in each direction, or local Nx and Ny
    int rem;
    
    int worldRank, size;    
    
    //return rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));   //round sqrt to nearest whole number
    xDomainSize = globalNx / p;           //minimum size of each process in x and y domain
    yDomainSize = globalNy / p;

    //first assign for y dimension
    rem = globalNy % p;                   //remainder, denotes how many processes need an column in domain
    int gridRank;
    int dims = 2;
    int coords[2];
    MPI_Comm_rank(grid, &gridRank);         //retrieve rank in grid, also check if grid created successfully
    MPI_Cart_coords(grid, gridRank, dims, coords);        //generate coordinates
    
    if(coords[0] < rem) {//safer to use coordinates (row) than rank, which could be reordered, if coord(row)< remainder, use minimum + 1
        yDomainSize++;
        yStart = yDomainSize * coords[0];             //index denoting starting row in local domain
    }
    else {//otherwise use minimum, and find other values
        yStart = (yDomainSize + 1) * rem + yDomainSize * (coords[0] - rem);           //starting row accounts for previous processes with +1 rows and +0 rows
    }
    
    //same for x dimension
    rem = globalNx % p;
        
    if(coords[1] < rem) {//safer to use coordinates (column) than rank, which could be reordered, if coord(column)< remainder, use minimum + 1
        xDomainSize++;
        xStart = xDomainSize * coords[1];             //index denoting starting column in local domain
    }
    else {//otherwise use minimum, and find other values
        xStart = (xDomainSize + 1) * rem + xDomainSize * (coords[1] - rem);           //starting column accounts for previous processes with +1 rows and +0 rows
    }
    
    localNx = xDomainSize;
    localNy = yDomainSize;
    localLx = (double) globalLx * localNx / globalNx;
    localLy = (double) globalLy * localNy / globalNy;
}

/**
 * @brief Test SolverCG constructor is assigning values correctly
 */
BOOST_AUTO_TEST_CASE(SolverCG_Constructor)
{

    const int Nx = 100;
    const int Ny = 50;
    const double dx = 0.05;
    const double dy = 0.02;
    
    //set up MPI for solver and split domain equally
    MPI_Comm grid,row,col;
    int localNx = 0;
    int localNy = 0;
    int ignore;
    double ignoreDouble;

    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, ignoreDouble,ignoreDouble,localNx,localNy,ignoreDouble,ignoreDouble,ignore,ignore);

    //Each local SolverCG should have localNx, localNy, as that is the defined behaviour
    SolverCG test(localNx,localNy,dx,dy,row,col);
    
    int testLocalNx = test.GetNx();
    int testLocalNy = test.GetNy();
    int testGlobalNx,testGlobalNy;
    MPI_Allreduce(&testLocalNx,&testGlobalNx,1,MPI_INT,MPI_SUM,row);
    MPI_Allreduce(&testLocalNy,&testGlobalNy,1,MPI_INT,MPI_SUM,col);//compute total 
    
    BOOST_CHECK_EQUAL(test.GetNx(),localNx);
    BOOST_CHECK_EQUAL(test.GetNy(),localNy);
    BOOST_CHECK_EQUAL(testGlobalNx,Nx);
    BOOST_CHECK_EQUAL(testGlobalNy,Ny);
    BOOST_CHECK_CLOSE(test.GetDx(),dx,1e-6);
    BOOST_CHECK_CLOSE(test.GetDy(),dy,1e-6);
}

/**
 * @brief Test SolverCG::Solve where if input b is very close to zero, then output x should be exactly 0.0 for all entries
 */
BOOST_AUTO_TEST_CASE(SolverCG_NearZeroInput)
{
    
    const int Nx = 10;                                      //define grid and steps
    const int Ny = 10;
    double dx = 0.1;
    double dy = 0.1;    
    
    //set up MPI for solver and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy,ignore;
    double ignoreDouble;
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, ignoreDouble, ignoreDouble, localNx,localNy,ignoreDouble,ignoreDouble,ignore,ignore);
    int n = localNx*localNy;                                //total number of grid points in process

    SolverCG test(localNx,localNy,dx,dy,row,col);           //create test solver

    double *b = new double[n];                              //allocate memory of input b and output x, denotes equation Ax = b
    double *x = new double[n];
    
    for(int i = 0; i < n; i++) {
        b[i] = 1e-8;                                        //100 element array with each element = 1e-8
    }                                                       //2-norm of b is smaller than tol*tol where tol = 1e-3 as specified in SolverCG 

    //pass data through solver, each process should solve part of the problem
    test.Solve(b,x);                                        //Solve Ax=b for x
    
    for(int i = 0; i < n; i++) {                            //no need to collect, each term should be 0
        BOOST_CHECK(x[i]-0.0<1e-20);
    }
    
    delete[] b;                                             //deallocate memory
    delete[] x;
}

/** 
 * @brief Sinusoidal test case Ax=b. Since A is the coefficients of the operator \f$ -\nabla^2 \f$, then the sinusoidal test case is
 * \f$ - \pi ^2 (k^2 + l^2) \sin(k \pi x) \sin (l \pi y) \f$, on a domain \f$ (x,y) \in [0, \frac{2}{k}] \times [0, \frac{2}{l}] \f$. 
 * Domain choice ensures zero boundary conditions on domain edge is imposed. The solution should satsify
 * \f$ x = - \sin (k \pi x) \sin (l \pi y) \f$. First guess x is generated randomly in domain \f$ [0,1] \f$, with zero boundary conditions imposed.
 */
BOOST_AUTO_TEST_CASE(SolverCG_SinusoidalInput) 
{    
    const int k = 3;                                    //sin(k*pi*x)sin(l*pi*y)
    const int l = 3;
    const double Lx = 2.0 / k;                          //correct domain for problem, such that sin sin has zero boundary conditions
    const double Ly = 2.0 / l;
    const int Nx = 2000;                                //define number of grids with correct step sizes, this is for the entire 'global' domain
    const int Ny = 2000;
    double dx = (double)Lx/(Nx - 1);
    double dy = (double)Ly/(Ny - 1);    
    double tol = 1e-3;                                  //define the tolerance

    //create the communicator that SolverCG expects
    MPI_Comm grid,row,col;
    int localNx,localNy,xStart,yStart;
    double ignoreDouble;
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,ignoreDouble,ignoreDouble,xStart,yStart);    //compute local domain of each process
    int n = localNx*localNy;                            //local number of points in process

    double *b = new double[n];                          //allocate memory
    double *x = new double[n];
    double* x_actual = new double[n];

    SolverCG test(localNx,localNy,dx,dy,row,col);       //create test solver
    
    for(int i = 0; i < n; i++) {//other ways to do this
        b[i] = 0.0;                                     //initialise b and x with zeros
        x[i] = 0.0;                                     //zero BCs naturally satisfied, zeros also improve convergence speed
    }
    for (int i = xStart; i < xStart+localNx; ++i) {                      //generate the sinusoidal test case input b, give correct chunk to each process
        for (int j = yStart; j < yStart + localNy; ++j) {
            b[IDX(i-xStart,j-yStart)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    
    //each process should solve part of the problem Ax=b
    test.Solve(b,x);                                    //Solve the sinusoidal test case

    for(int i = xStart; i < xStart + localNx; ++i){                        //Generate the analytical solution x for each chunk
        for(int j = yStart; j < yStart + localNy; ++j) {
            x_actual[IDX(i-xStart,j-yStart)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }

    cblas_daxpy(n, -1.0, x, 1, x_actual, 1);             //compute error between analytical and solver, store in x_actual

    double e = cblas_dnrm2(n,x_actual,1);                //2-norm error, to reduce local to global, need to sum squares of error for linearity
    e *= e;
    double globalError;
    MPI_Allreduce(&e,&globalError,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    globalError = sqrt(globalError);                    //summed error squared, so use errror

    BOOST_CHECK(globalError < tol);                     //check the error 2-norm is smaller than tol*tol, or 1e-3

    delete[] x;                                         //deallocate memory
    delete[] x_actual;
    delete[] b;
}

/**
 * @brief Tests whether LidDrivenCavity constructor is generated correctly in MPI implementation. Should split the default domain in unlikely case that it is used
*/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_Constructor) {
    //default values in class, global values of problem
    int Nx = 9;
    int Ny = 9;
    int Npts = 81;
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = (double)Lx/(Nx-1);
    double dy = (double)Ly/(Ny-1);
    double dt = 0.01;
    double T = 1.0;
    double Re = 10.0;
    double U = 1.0;
    double nu = 0.1;

    //MPI implementation
    MPI_Comm grid,row,col;
    int localNx,localNy,ignore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,ignore,ignore);//default Nx, Ny is 9; should be split by constructor into localNx and localNy
    
    LidDrivenCavity test;

    //these values should be the same across all processes
    double tol = 1e-3;
    BOOST_CHECK_CLOSE(test.GetDt(), dt, tol);
    BOOST_CHECK_CLOSE(test.GetT(), T, tol);
    BOOST_CHECK_CLOSE(test.GetRe(), Re, tol);
    BOOST_CHECK_CLOSE(test.GetU(), U, tol);
    BOOST_CHECK_CLOSE(test.GetNu(),nu,tol);
    BOOST_CHECK_CLOSE(test.GetDx(),dx,tol);
    BOOST_CHECK_CLOSE(test.GetDy(),dy,tol);

    //check local and global chunks
    BOOST_REQUIRE(test.GetNx() == localNx);//check local values assigned to each chunk
    BOOST_REQUIRE(test.GetNy() == localNy);
    BOOST_REQUIRE(test.GetNpts() == localNx*localNy);
    
    BOOST_REQUIRE(test.GetLx() == localLx);
    BOOST_REQUIRE(test.GetLy() == localLy);

    BOOST_REQUIRE(test.GetGlobalNx() == Nx);//check correct global value
    BOOST_REQUIRE(test.GetGlobalNy() == Ny);
    BOOST_REQUIRE(test.GetGlobalNpts() == Npts);

    BOOST_REQUIRE(test.GetGlobalLx() == Lx);
    BOOST_REQUIRE(test.GetGlobalLy() == Ly);
}

/**
 * @brief Test whether LidDrivenCavity::SetDomainSize assigns values correctly and correctly configures problem
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetDomainSize) {
    
    //default values in class
    int Nx = 9;
    int Ny = 9;
    int Npts = 81;

    //values to assign and expectations
    double Lx = 2.2;
    double Ly = 3.3;
    double dx = (double)Lx/(Nx-1);
    double dy = (double)Ly/(Ny-1);

    //set up MPI communicators and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy,ignore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,ignore,ignore);//default Nx, Ny is 9; should be split by constructor

    LidDrivenCavity test;         //lid driven cavity with default values
    
    test.SetDomainSize(Lx,Ly);          //call function to be tested, input global values
    
    //check whether values changed correctly
    double tol = 1e-6;
    BOOST_REQUIRE( abs(test.GetLx() - localLx) < tol);//check local values
    BOOST_REQUIRE( abs(test.GetLy() - localLy) < tol);
    BOOST_REQUIRE( abs(test.GetDx() - dx) < tol);
    BOOST_REQUIRE( abs(test.GetDy() - dy) < tol);
    BOOST_REQUIRE(test.GetNx() == localNx);
    BOOST_REQUIRE(test.GetNy() == localNy);
    BOOST_REQUIRE(test.GetNpts() == localNx*localNy);

    BOOST_REQUIRE( abs(test.GetGlobalLx() - Lx) < tol);//check global values
    BOOST_REQUIRE( abs(test.GetGlobalLy() - Ly) < tol);
    BOOST_REQUIRE( test.GetGlobalNx() == Nx);
    BOOST_REQUIRE( test.GetGlobalNy() == Ny);
    BOOST_REQUIRE( test.GetGlobalNpts() == Npts);
}

/**
 * @brief Test whether LidDrivenCavity::SetGridSize assigns values correctly and correctly configures problem
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetGridSize) {
    
    //default values in class
    double Lx = 1.0;
    double Ly = 1.0;

    //values to assign and what expected values are
    int Nx = 102;
    int Ny = 307;                        //grid points in x and y
    int Npts = Nx*Ny;
    double dx = (double)Lx/(Nx-1);
    double dy = (double)Ly/(Ny-1);

    //set up MPI communicators and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy,ignore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,ignore,ignore);//default Nx, Ny is 9; should be split by constructor
    
    LidDrivenCavity test;

    test.SetGridSize(Nx,Ny);          //call function to be tested -> pass global value
    
    //check whether values changed correctly
    double tol = 1e-6;
    BOOST_REQUIRE( abs(test.GetLx() - localLx) < tol);//check local values
    BOOST_REQUIRE( abs(test.GetLy() - localLy) < tol);
    BOOST_REQUIRE( abs(test.GetDx() - dx) < tol);
    BOOST_REQUIRE( abs(test.GetDy() - dy) < tol);
    BOOST_REQUIRE(test.GetNx() == localNx);
    BOOST_REQUIRE(test.GetNy() == localNy);
    BOOST_REQUIRE(test.GetNpts() == localNx*localNy);

    BOOST_REQUIRE( abs(test.GetGlobalLx() - Lx) < tol);//check global values
    BOOST_REQUIRE( abs(test.GetGlobalLy() - Ly) < tol);
    BOOST_REQUIRE( test.GetGlobalNx() == Nx);
    BOOST_REQUIRE( test.GetGlobalNy() == Ny);
    BOOST_REQUIRE( test.GetGlobalNpts() == Npts);
}

/**
 * @brief Test whether LidDrivenCavity::SetTimeStep assigns values correctly and correctly configures problem
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetTimeStep) {

    //variable to set
    double dt = 0.024;

    //MPI implementation
    
    LidDrivenCavity test;

    test.SetTimeStep(dt);          //call function to be tested
    
    //no other expected values, check dt
    double tol = 1e-6;
    BOOST_REQUIRE( abs(test.GetDt() - dt) < tol);
}

/**
 * @brief Test whether LidDrivenCavity::SetFinalTime assigns values correctly and correctly configures problem
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetFinalTime) {

    //values to assign
    double T = 23.43;
    
    LidDrivenCavity test;

    test.SetFinalTime(T);          //call function to be tested
    
    //no other expected values, check T only
    double tol = 1e-6;
    BOOST_REQUIRE( abs(T - test.GetT()) < tol);

}

/**
 * @brief Test whether LidDrivenCavity::SetReynoldsNumber assigns values correctly and correctly configures problem
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetReynoldsNumber) {
    //values to assign
    double Re = 5000;

    //expected values
    double U = 1.0;
    double nu = U/Re;
    
    LidDrivenCavity test;

    test.SetReynoldsNumber(Re);          //call function to be tested
    
    //check whether changed values,  is actually changed correctly
    double tol = 1e-6;
    BOOST_REQUIRE( abs(Re - test.GetRe()) < tol);
    BOOST_REQUIRE( abs(nu - test.GetNu()) < tol);
    BOOST_REQUIRE( abs(U - test.GetU()) < tol);
}

/**
 * @brief Test case to confirm whether LidDrivenCavity::PrintConfiguration function prints out the correct configuration
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_PrintConfiguration)
{
    //define a test case with different numbers for each
    double dt   = 0.2;                          //time step
    double T    = 5.1;                          //final time
    int    Nx   = 21;
    int    Ny   = 11;
    //int    Npts = 231;                        //number of grid points, for reference
    double Lx   = 1.0;                          //domain length in x and y direction
    double Ly   = 2.0;
    double Re   = 100;                          //Reynolds number
    //double U    = 1.0;                        //flow speed, for reference
    //double nu   = 0.01;                       //kinematic viscosity, for reference
    //double dx = 0.05;                         //step size in x and y, for reference
    //double dy = 0.2;

    //note that nu * dt / dx / dy = 0.2 < 0.25, so solver should not exit
    
    //expected string outputs, taking care that spacing is same as that from output, and formatting of numbers is same as cout
    string configGridSize = "Grid size: 21 x 11";
    string configSpacing = "Spacing:   0.05 x 0.2";
    string configLength = "Length:    1 x 2";
    string configGridPts = "Grid pts:  231";
    string configTimestep = "Timestep:  0.2";
    string configSteps = "Steps:     26";           //also test ability of ceiling, as 25.5 should round up to 26
    string configReynolds = "Reynolds number: 100";
    string configOther = "Linear solver: preconditioned conjugate gradient";

    //MPI implementation
    MPI_Comm grid,row,col;
    CreateCartGridVerify(grid,row,col);

    int rowRank,colRank;
    MPI_Comm_rank(row,&rowRank);
    MPI_Comm_rank(col,&colRank);      //need to know as printing should only occur on root rank, or rowRank = 0 and colRank = 0

    LidDrivenCavity test;
    
    // Redirect cout to a stringstream for capturing the output
    std::stringstream terminalOutput;
    std::streambuf* oldCout = std::cout.rdbuf(terminalOutput.rdbuf());

    // Invoke setting functions and print the solver configuration to stringstream
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    test.PrintConfiguration();

    // Restore the original cout
    std::cout.rdbuf(oldCout);

    // Perform Boost Test assertions on the expected output, if fails, then no point continuing with other tests
    //check if desired string is present in the terminal output
    std::string output = terminalOutput.str();

    //root rank should print
    if((rowRank == 0) & (colRank == 0)) {
        BOOST_REQUIRE(output.find(configGridSize) != std::string::npos);
        BOOST_REQUIRE(output.find(configSpacing) != std::string::npos);
        BOOST_REQUIRE(output.find(configLength) != std::string::npos);
        BOOST_REQUIRE(output.find(configGridPts) != std::string::npos);
        BOOST_REQUIRE(output.find(configTimestep) != std::string::npos);
        BOOST_REQUIRE(output.find(configSteps) != std::string::npos);
        BOOST_REQUIRE(output.find(configReynolds) != std::string::npos);
        BOOST_REQUIRE(output.find(configOther) != std::string::npos);
    }
    else {  //no other ranks should do so, but not critical

        BOOST_CHECK(output.find(configGridSize) == std::string::npos);
        BOOST_CHECK(output.find(configSpacing) == std::string::npos);
        BOOST_CHECK(output.find(configLength) == std::string::npos);
        BOOST_CHECK(output.find(configGridPts) == std::string::npos);
        BOOST_CHECK(output.find(configTimestep) == std::string::npos);
        BOOST_CHECK(output.find(configSteps) == std::string::npos);
        BOOST_CHECK(output.find(configReynolds) == std::string::npos);
        BOOST_CHECK(output.find(configOther) == std::string::npos);
    }
}

/**
 * @brief Test whether LidDrivenCavity::Initialise initialises the vorticity, streamfunctions and velocities correctly
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_Initialise) {
    //take previous working test case, same variable definitions as before
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;
    //double dx = 0.05;
    //double dy = 0.2;

   //split the domain and compute what each MPI communicator data should be for initialise
    MPI_Comm grid,row,col;
    int localNx,localNy,xStart,yStart;
    double ignoreDouble;
    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,ignoreDouble,ignoreDouble,xStart,yStart);    //compute local domain of each process
    int localNpts = localNx*localNy;                            //local number of points in process

    //compute ranks for adjacent grids for data transfer, if at boundary, returns -2 (MPI_PROC_NULL)
    int bottomRank,topRank;
    MPI_Cart_shift(col,0,1,&bottomRank,&topRank);//from bottom to top

    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    test.Initialise();                              //initialise the problem
    
    double* v = new double[localNpts];
    double* s = new double[localNpts];//v and s should be zero
    
    test.GetData(v,s);

    //initialise implies zeros for all values, so norm v and norm s give the error norm
    double tol = 1e-6;
    double globalErrorV,globalErrorS;
    double eV = cblas_dnrm2(localNpts,v,1);
    double eS = cblas_dnrm2(localNpts,s,1);
    eV *= eV;
    eS *= eS;

    MPI_Allreduce(&eV,&globalErrorV,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&eS,&globalErrorS,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    globalErrorV = sqrt(globalErrorV);
    globalErrorS = sqrt(globalErrorS);

    BOOST_CHECK(globalErrorS < tol);
    BOOST_CHECK(globalErrorV < tol);

    delete[] v;
    delete[] s;
}

/**
 * @brief Test whether LidDrivenCavity::WriteSolution() creates file and outputs correct data in correct format. Uses initial condition data to check.
 * Upon problem initialisation, streamfunction and voriticity should be zero everywhere. Vertical and horizontal velocities should be zero 
 * everywhere, except at top of lid where horizontal velocity is 1.
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_WriteSolution) 
{
    //take previous working test case, same variable definitions as before
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;
    double U = 1;
    double dx = 0.05;
    double dy = 0.2;

    int localNx = Nx;//so that IDX can be used
    
    //expect the following values, while vy, s and v should all be zero
    double* x = new double[Nx*Ny];
    double* y = new double[Nx*Ny];
    double* vx = new double[Nx*Ny]();

    for(int i = 0; i < Nx; ++i) {
        for(int j = 0; j < Ny; ++j) {
            x[IDX(i,j)] = i*dx;
            y[IDX(i,j)] = j*dy;
        }
    }

    for(int i = 0; i < Nx; ++i) {
        vx[IDX(i,Ny-1)] = U;
    }

    //set up MPI for solver and split domain equally
    int worldRank; 
    MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);//get world rank

    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    test.Initialise();                              //initialise the problem
    std::string fileName = "testOutput";            //output initial conditions to file named testOutput
    test.WriteSolution(fileName);                   //only produces one file, even in MPI

    std::ifstream outputFile(fileName);             //create stream for file, reading data across multiple processors is okay as they have independent access
    BOOST_REQUIRE(outputFile.is_open());            //check if file has been created by seeing if it can be opened, if doesn't exist, terminate
    
    //read data from file and check initial conditions -> verifies WriteSolution();
    //expect format of (x,y), (vorticity, streamfunction), (vx,vy)
    //initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have vx = 1
    
    std::string line;                                               //variable to capture line of data from file
    std::string xData,yData,vData,sData,vxData,vyData;              //temporary string variables
    
    double* xDataSet = new double[Nx*Ny];
    double* yDataSet = new double[Nx*Ny];
    double* vDataSet = new double[Nx*Ny];
    double* sDataSet = new double[Nx*Ny];
    double* vxDataSet = new double[Nx*Ny];
    double* vyDataSet = new double[Nx*Ny];

    int i = 0;                                                      //denote matrix index i,j
    int j = 0;
    int dataPoints = 0;                     //counter to track number of data points printed, should equal Nx*Ny
    unsigned int dataCol = 0;               //conter for number of columns per row, should be 6

    while(std::getline(outputFile,line)) {          //while file is open
        
        if(line.empty()) {
            continue;                               //if empty line in file, skip
        }
        
        //first check that there are six datapoints per row
        std::stringstream dataCheck(line); 
        
        while (dataCheck >> xData) {
                dataCol++;                              //calculate how many data points per line
        }
        
        BOOST_REQUIRE(dataCol == 6);                    //check data printed correctly, 6 data points per row
        dataCol = 0;                                    //reset data counter
        
        //now read data into correct place
        std::stringstream data(line);
        data >> xData >> yData >> vData >> sData >> vxData >> vyData;

        xDataSet[IDX(i,j)] = std::stod(xData);
        yDataSet[IDX(i,j)] = std::stod(yData);
        vDataSet[IDX(i,j)] = std::stod(vData);
        sDataSet[IDX(i,j)] = std::stod(sData);
        vxDataSet[IDX(i,j)] = std::stod(vxData);
        vyDataSet[IDX(i,j)] = std::stod(vyData);
        
        dataPoints++;                           //log one more data point
        
        //also increment i and j counters; note that data is written out column by column, so i constant, j increments
        j++;
        if(j >= Ny) {       //if j exceeds number of rows (or Ny), then we want index to point to the start of next column 
            j = 0;
            i++;
        }
    }
    outputFile.close();                          //close the file

    BOOST_CHECK(dataPoints == Nx*Ny);            //check right number of grid point data is outputted
    
    //compute global errors
    cblas_daxpy(Nx*Ny,-1.0,x,1,xDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,y,1,yDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,vx,1,vxDataSet,1);    //reference for all other variables is 0, so no need to compute error vector

    double xError = cblas_dnrm2(Nx*Ny,xDataSet,1);
    double yError = cblas_dnrm2(Nx*Ny,yDataSet,1);
    double vxError = cblas_dnrm2(Nx*Ny,vxDataSet,1);
    double vyError = cblas_dnrm2(Nx*Ny,vyDataSet,1);
    double sError = cblas_dnrm2(Nx*Ny,sDataSet,1);
    double vError = cblas_dnrm2(Nx*Ny,vDataSet,1);

    //check within tolerance
    double tol = 1e-6;
    BOOST_CHECK(xError < tol);
    BOOST_CHECK(yError < tol);
    BOOST_CHECK(vxError < tol);
    BOOST_CHECK(vyError < tol);
    BOOST_CHECK(sError < tol);
    BOOST_CHECK(vError < tol);

    delete[] xDataSet;
    delete[] yDataSet;
    delete[] vDataSet;
    delete[] sDataSet;
    delete[] vxDataSet;
    delete[] vyDataSet;
    delete[] x;
    delete[] y;
    delete[] vx;

    /*if(std::remove(fileName.c_str()) == 0) {      //use make clean to delete the test file
        std::cout << fileName << " successfully deleted" << std::endl;
    }
    else {
        std::cout << fileName << " could not be deleted" << std::endl;
    }/**/
}

/**
 * @brief Tests whether the time domain solver LidDrivenCavity::Integrator works correctly by computing 5 time steps for a specified problem.
 * Solution is compared to the output generated by the serial solver, whose data is stored in a text file named "DataIntegratorTestCase".
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_Integrator) 
{
    //take a case where steady state is reached -> rule of thumb, fluid should pass through at least 10 times to reach SS
    double dt   = 0.005;
    double T    = 20;//0.01;
    int    Nx   = 201;//5;//10;
    int    Ny   = 201;//5;//10;
    double Lx   = 1;
    double Ly   = 1;
    double Re   = 1000;
    //double dx = 0.005;                     //for reference, step sizes
    //double dy = 0.005;
    //nu * dt/dx/dy = 0.2 < 0.25            //for reference, check the step size limit is smaller than 0.25
       
    int localNx = Nx;//so that IDX can be used
    
    //expect the following values array
    double* x = new double[Nx*Ny];
    double* y = new double[Nx*Ny];
    double* vx = new double[Nx*Ny];
    double* vy = new double[Nx*Ny];
    double* s = new double[Nx*Ny];
    double* v = new double[Nx*Ny];

    //store new results
    double* xDataSet = new double[Nx*Ny];
    double* yDataSet = new double[Nx*Ny];
    double* vDataSet = new double[Nx*Ny];
    double* sDataSet = new double[Nx*Ny];
    double* vxDataSet = new double[Nx*Ny];
    double* vyDataSet = new double[Nx*Ny];

    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    //initialise and write the output
    test.Initialise();
    test.Integrate();
    //exit(-1);
    //output file containing initial condition, in file name IntegratorTest; reference data in DataIntegratorTestCase
    std::string fileName = "IntegratorTest";
    std::string refData = "DataIntegratorTestCase";
    test.WriteSolution(fileName);
    
    std::ifstream outputFile(fileName);             // Create streams for outputted data and reference data
    std::ifstream refFile(refData);
    
    BOOST_REQUIRE(outputFile.is_open());            //check if output file has been created by seeing if it can be opened
    BOOST_REQUIRE(refFile.is_open());               //check if reference data file exists by seeing if it can be opened

    //read data from file and compare with reference data from serial solver
    //expect format of (x,y), (vorticity, streamfunction), (vx,vy)
    //initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have vx = 1
    
    std::string line,refLine;                                   //variable to capture line of data from file
    std::string xData,yData,vData,sData,vxData,vyData;          //temporary string variables
    
    int i = 0;
    int j = 0;
    int dataPoints = 0;                                         //counter to track number of data points printed, should equal Nx*Ny
    unsigned int dataCol = 0;                                   //conter for number of columns per row, should be 6
    //while file is open
    while(std::getline(outputFile,line)) {
        
        std::getline(refFile,refLine);                          //get data from ref file
        if(line.empty()) {
            continue;                                           //if empty line in file, skip
        }
        
        //first check that there are six pieces of data per row
        std::stringstream dataCheck(line);
        
        while (dataCheck >> xData) {
                dataCol++;
        }
        BOOST_REQUIRE(dataCol == 6);                            //check data printed correctly, 6 data points per row
        dataCol = 0;                                            //reset data counter
        
        //now read outputted data and reference data from serial solver into correct place
        std::stringstream data(line);
        std::stringstream dataRef(refLine);
        
        data >> xData >> yData >> vData >> sData >> vxData >> vyData;
        
        xDataSet[IDX(i,j)] = std::stod(xData);
        yDataSet[IDX(i,j)] = std::stod(yData);
        vDataSet[IDX(i,j)] = std::stod(vData);
        sDataSet[IDX(i,j)] = std::stod(sData);
        vxDataSet[IDX(i,j)] = std::stod(vxData);
        vyDataSet[IDX(i,j)] = std::stod(vyData);
        
        dataRef >> xData >> yData >> vData >> sData >> vxData >> vyData;
        
        x[IDX(i,j)] = std::stod(xData);
        y[IDX(i,j)] = std::stod(yData);
        v[IDX(i,j)] = std::stod(vData);
        s[IDX(i,j)] = std::stod(sData);
        vx[IDX(i,j)] = std::stod(vxData);
        vy[IDX(i,j)] = std::stod(vyData);
        
        dataPoints++;                           //log extra data pionts

        //also increment i and j counters; note that data is written out column by column, so i constant, j increments
        j++;
        if(j >= Ny) {       //if j exceeds number of rows (or Ny), then we want index to point to the start of next column 
            j = 0;
            i++;
        }
    }
    outputFile.close();                         //close files
    refFile.close();
    
    BOOST_CHECK(dataPoints == Nx*Ny);            //check right number of grid point data is outputted

    //compute global errors
    cblas_daxpy(Nx*Ny,-1.0,x,1,xDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,y,1,yDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,vx,1,vxDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,vy,1,vyDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,s,1,sDataSet,1);
    cblas_daxpy(Nx*Ny,-1.0,v,1,vDataSet,1);

    //compute errors norm
    double xError = cblas_dnrm2(Nx*Ny,xDataSet,1);
    double yError = cblas_dnrm2(Nx*Ny,yDataSet,1);
    double vxError = cblas_dnrm2(Nx*Ny,vxDataSet,1);
    double vyError = cblas_dnrm2(Nx*Ny,vyDataSet,1);
    double sError = cblas_dnrm2(Nx*Ny,sDataSet,1);
    double vError = cblas_dnrm2(Nx*Ny,vDataSet,1);

    
        cout << "xError = " << xError << endl;
        cout << "yError = " << yError << endl;
        cout << "vxError = " << vxError << endl;
        cout << "vyError = " << vyError << endl;
        cout << "vError = " << vError << endl;
        cout << "sError = " << sError << endl;
    

    //check within tolerance
    double tol = 1e-3;
    BOOST_CHECK(xError < tol);
    BOOST_CHECK(yError < tol);
    BOOST_CHECK(vxError < tol);
    BOOST_CHECK(vyError < tol);
    BOOST_CHECK(sError < tol);
    BOOST_CHECK(vError < tol);

    delete[] xDataSet;
    delete[] yDataSet;
    delete[] vDataSet;
    delete[] sDataSet;
    delete[] vxDataSet;
    delete[] vyDataSet;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    delete[] s;
    delete[] v;

    //delete output file from directory to clean up with make clean
    /*if(std::remove(fileName.c_str()) == 0) {
        std::cout << fileName << " successfully deleted" << std::endl;
    }
    else {
        std::cout << fileName << " could not be deleted" << std::endl;
    }*/
}