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
 * @brief Macro to map coordinates (i,j) onto it's corresponding location in memory, assuming row-wise matrix storage
 * @param I     coordinate i denoting horizontal position of grid from left to right
 * @param J     coordinate j denoting vertical position of grid from bottom to top
 */
#define IDX(I,J) ((J)*localNx + (I))

/**
 * @brief Allow MPI to be initialised and finalised once throughout the unit test
*/
struct MPISetUp {
    /**
     * @brief Initialise MPI
    */
    MPISetUp() {
        // Access argc and argv from Boost Test framework
        int& argc = boost::unit_test::framework::master_test_suite().argc;
        char**& argv = boost::unit_test::framework::master_test_suite().argv;

        MPI_Init(&argc, &argv);
    }
    /**
     * @brief Finalise MPI
    */
    ~MPISetUp() {
        MPI_Finalize();
    }
};

BOOST_GLOBAL_FIXTURE(MPISetUp);

/**
 * @brief Create Cartesian topology grid and column and row communicators
 * @param[out] comm_Cart_Grid   Communicator for Cartesian grid
 * @param[out] comm_row_grid    Communicator for current row of Cartesian grid
 * @param[out] comm_col_grid    Communicator for current column of Cartesian grid
 *****************************************************************************************************************************/
void CreateCartGridVerify(MPI_Comm &comm_Cart_Grid,MPI_Comm &comm_row_grid, MPI_Comm &comm_col_grid){
    
    int worldRank, size;    
    
    //return rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));                                          //round sqrt to nearest whole number
    
    if((p*p != size) | (size < 1)) {                                    //if not a square number, print error and terminate program
        if(worldRank == 0)                                              //print only on root rank
            cout << "Invalide process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        exit(-1);
    }

    /* Set up Cartesian topology to represent the 'grid' domain of the lid driven cavity problem
    Treat root process as bottom left of grid, with Cartesian coordinates (i,j)
    Increasing i goes to the right and increasing j goes up*/

    const int dims = 2;                                                                     //2 dimensions in grid
    int gridSize[dims] = {p,p};                                                             //p processes per dimension
    int periods[dims] = {0,0};                                                              //grid is not periodic
    int reorder = 1;                                                                        //reordering of grid allowed
    int keep[dims];                                                                         //denotes which dimension to keep when finding subgrids

    MPI_Cart_create(MPI_COMM_WORLD,dims,gridSize,periods,reorder, &comm_Cart_Grid);         //create Cartesian topology grid
    
    //create row communnicator in subgrid so process can communicate with other processes on row   
    keep[0] = 0;        
    keep[1] = 1;                                                        //keep all processes with same j coordinate i.e. same row
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_row_grid);
    
    //create column communnicator in subgrid so process can communicate with other processes on column
    keep[0] = 1;        
    keep[1] = 0;                                                        //keep all processes with same i coordinate i.e. same column
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_col_grid);
}

/**
 * @brief Split the global lid driven cavity domain into local grid cavity for each MPI process
 * @param[in] grid      MPI communicator denoting a Cartesian topology grid
 * @param[in] globalNx  The number of grid points in the x direction in the global lid driven cavity domain
 * @param[in] globalNy  The number of grid points in the y direction in the global lid driven cavity domain
 * @param[in] globalLx  The length in the x direction of the global lid driven cavity domain
 * @param[in] globalLy  The length in the y direction of the global lid driven cavity domain
 * @param[out] localNx  The number of grid points in the x direction in the local lid driven cavity domain of an MPI process
 * @param[out] localNy  The number of grid points in the y direction in the local lid driven cavity domain of an MPI process
 * @param[out] localLx  The length in the x direction of the local lid driven cavity domain of an MPI process
 * @param[out] localLx  The length in the x direction of the local lid driven cavity domain of an MPI process
 * @param[out] xStart   Starting point of local domain in global domain, x direction
 * @param[out] yStart   Starting point of local domain in global domain, y direction
 * @warning MPI ranks must satisfy \f$ P = p^2 \f$, otherwise program will terminate
 ****************************************************************************************************************************/
void SplitDomainMPIVerify(MPI_Comm &grid, int globalNx, int globalNy, double globalLx, double globalLy, 
                        int &localNx, int &localNy, double &localLx, double &localLy, int &xStart, int &yStart) {
    
    /*First generate some useful data from MPI
    Ranks could be reordered in Cartesian grid, so coordinates are used instead of rank to ensure domain is split appropriately
    If reordered ranks are used, then all processes along a row may not have the same y domain size -> will kill the program
    Coordinates will always be in the place you expect them to be, while rank may not be due to reordering*/
    int rem,size,gridRank; 
    int dims = 2;
    int coords[dims];

    MPI_Comm_size(MPI_COMM_WORLD, &size);                       //return total number of MPI ranks, size denotes total number of processes P
    MPI_Comm_rank(grid, &gridRank);
    MPI_Cart_coords(grid, gridRank, dims, coords);              //use process rank in Cartesian grid to generate coordinates
    
    //assume that P = p^2 is already verified and find p, the number of processes along each domain dimension
    int p = round(sqrt(size));
    localNx = globalNx / p;                                     //minimum local size x and y domain for each process
    localNy = globalNy / p;

    //first assign for y dimension
    rem = globalNy % p;                                         //remainder denotes how many processes need to take an extra grid point in y direction (or row)

    if(coords[0] < rem) {                                       //add 1 extra row to first rem processes
        localNy++;
        yStart = localNy * coords[0];                           //index denoting how the starting row of the local domain maps onto the global domain
    }
    else {
        yStart = (localNy + 1) * rem + localNy * (coords[0] - rem);      //starting row accounts for previous processes with +1 rows and +0 rows
    }
    
    //same logic for x dimension (same as above, replacing "row" with "column" and "y" with "x")
    rem = globalNx % p;
        
    if(coords[1] < rem) {
        localNx++;
        xStart = localNx * coords[1];
    }
    else {
        xStart = (localNx + 1) * rem + localNx * (coords[1] - rem);
    }
    
    localLx = (double) globalLx * localNx / globalNx;           //compute local domain length by considering ratio of local domain size to global domain size
    localLy = (double) globalLy * localNy / globalNy;
}

/**
 * @brief Test SolverCG constructor is assigning values correctly
 ******************************************************************************************************************/
BOOST_AUTO_TEST_CASE(SolverCG_Constructor)
{
    //global domain variables
    const int Nx = 100;
    const int Ny = 50;
    const double dx = 0.05;
    const double dy = 0.02;
    
    //set up MPI for solver and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy;                                                            //denote the local domain sizes of each process
    int iIgnore;
    double dIgnore;

    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, dIgnore,dIgnore,localNx,localNy,dIgnore,dIgnore,iIgnore,iIgnore);

    //Each local SolverCG should have localNx, localNy, as that is the defined behaviour
    SolverCG test(localNx,localNy,dx,dy,row,col);
    
    int testLocalNx = test.GetNx();                                                 //grab local domain values
    int testLocalNy = test.GetNy();
    int testGlobalNx,testGlobalNy;

    MPI_Allreduce(&testLocalNx,&testGlobalNx,1,MPI_INT,MPI_SUM,row);
    MPI_Allreduce(&testLocalNy,&testGlobalNy,1,MPI_INT,MPI_SUM,col);                //compute total domain sizes
    
    //check local and global domain sizes are as expected; dx,dy should be the global values
    BOOST_CHECK_EQUAL(testLocalNx,localNx);
    BOOST_CHECK_EQUAL(testLocalNy,localNy);
    BOOST_CHECK_EQUAL(testGlobalNx,Nx);
    BOOST_CHECK_EQUAL(testGlobalNy,Ny);
    BOOST_CHECK_CLOSE(test.GetDx(),dx,1e-3);
    BOOST_CHECK_CLOSE(test.GetDy(),dy,1e-3);
}

/**
 * @brief Test SolverCG::Solve where if input b is very close to zero, then output x should be exactly 0.0 for all entries
 ************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(SolverCG_Solve_NearZeroInput)
{
    //define problem for global problem domain
    const int Nx = 10;                                      //define grid and steps
    const int Ny = 10;
    double dx = 0.1;
    double dy = 0.1;    
    
    //set up MPI for solver and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy;
    int iIgnore;
    double dIgnore;

    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, dIgnore, dIgnore, localNx,localNy,dIgnore,dIgnore,iIgnore,iIgnore);
    
    SolverCG test(localNx,localNy,dx,dy,row,col);           //create test solver

    //each process sets up the problem for its sub-domain
    int n = localNx * localNy;                              //total number of local grid points in process
    double *b = new double[n];                              //allocate memory of input b and output x, denotes equation Ax = b
    double *x = new double[n];
    
    for(int i = 0; i < n; i++) {
        b[i] = 1e-8;                                        //100 element array with each element = 1e-8
    }                                                       //2-norm of b is smaller than tol*tol where tol = 1e-3 as specified in SolverCG 

    //pass data through solver, each process should solve part of the problem
    test.Solve(b,x);                                        //Solve Ax=b for x
    
    for(int i = 0; i < n; i++) {                            //all terms should be 0, so can perform check with local processes
        BOOST_CHECK_SMALL(x[i],1e-6);
    }
    
    delete[] b;
    delete[] x;
}

/** 
 * @brief Sinusoidal test case for SolverCG which solves Ax=b. 
 * Since A is the coefficients of the operator \f$ -\nabla^2 \f$, then the sinusoidal test case is
 * \f$ - \pi ^2 (k^2 + l^2) \sin(k \pi x) \sin (l \pi y) \f$, on a domain \f$ (x,y) \in [0, \frac{2}{k}] \times [0, \frac{2}{l}] \f$. 
 * Domain choice ensures zero boundary conditions on domain edge is imposed. The solution should satsify
 * \f$ x = - \sin (k \pi x) \sin (l \pi y) \f$. First guess x is generated randomly in domain \f$ [0,1] \f$, with zero boundary conditions imposed.
 **************************************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(SolverCG_Solve_SinusoidalInput) 
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

    //create the communicators that SolverCG expects and split the domain
    MPI_Comm grid,row,col;
    int localNx,localNy,xStart,yStart;
    double dIgnore;

    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,dIgnore,dIgnore,xStart,yStart);    //compute local domain of each process
    
    //each process sets up the problem for its sub-domain
    int n = localNx*localNy;                            //local number of points in process
    double *b = new double[n]();                        //initialise with zeros, first guess x full of zeros improves convergence speed
    double *x = new double[n]();
    double* x_actual = new double[n]();

    SolverCG test(localNx,localNy,dx,dy,row,col);       //create test solver
    
    //generate the sinusoidal test case input b, make sure each process calculates the correct chunk and stores it in its local memory
    for (int i = xStart; i < xStart + localNx; ++i) {                                           //pluses to make sure each process calculates the correct value...
        for (int j = yStart; j < yStart + localNy; ++j) {                                       //that its local domain represents in the global domain
            b[IDX(i - xStart,j - yStart)] = -M_PI * M_PI * (k * k + l * l)                      //minus in index to ensure values are written...
                                       * sin(M_PI * k * i * dx)                                 //in correct places in the process' local array
                                       * sin(M_PI * l * j * dy);
        }
    }

    test.Solve(b,x);                                                                            //each process solves part of the problem Ax=b

    //Generate the analytical solution x for each chunk
    for(int i = xStart; i < xStart + localNx; ++i){                                             
        for(int j = yStart; j < yStart + localNy; ++j) {
            x_actual[IDX(i-xStart,j-yStart)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }

    //compute error between analytical and solver, store in x_actual
    cblas_daxpy(n, -1.0, x, 1, x_actual, 1);

    double e = cblas_dnrm2(n,x_actual,1);
    double globalError;

    e *= e;                                                             //2-norm error, to reduce local to global, need to sum squares of error for linearity
    MPI_Allreduce(&e,&globalError,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    globalError = sqrt(globalError);                                    //summed error squared, so use square root to get 2-norm of global error

    BOOST_CHECK(globalError < tol);

    delete[] x;
    delete[] x_actual;
    delete[] b;
}

/**
 * @brief Tests whether LidDrivenCavity constructor is generated correctly in MPI implementation. Should split the default domain in unlikely case that it is used
**************************************************************************************************************************************************************/
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
    int localNx,localNy,iIgnore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    //default Nx, Ny is 9; should be split by constructor into localNx and localNy; Lx and Ly should also be split into correct localLx and localLy
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,iIgnore,iIgnore);
    
    LidDrivenCavity test;

    //these values should be the same across all processes
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(test.GetDt(), dt, tol);
    BOOST_CHECK_CLOSE(test.GetT(), T, tol);
    BOOST_CHECK_CLOSE(test.GetRe(), Re, tol);
    BOOST_CHECK_CLOSE(test.GetU(), U, tol);
    BOOST_CHECK_CLOSE(test.GetNu(),nu,tol);
    BOOST_CHECK_CLOSE(test.GetDx(),dx,tol);
    BOOST_CHECK_CLOSE(test.GetDy(),dy,tol);

    //check local and global data stored by each process is correct
    BOOST_CHECK_EQUAL(test.GetNx(), localNx);                     //check local values assigned to each chunk
    BOOST_CHECK_EQUAL(test.GetNy(), localNy);
    BOOST_CHECK_EQUAL(test.GetNpts(), localNx*localNy);
    
    BOOST_CHECK_CLOSE(test.GetLx(),localLx,tol);
    BOOST_CHECK_CLOSE(test.GetLy(),localLy,tol);

    BOOST_CHECK_EQUAL(test.GetGlobalNx(), Nx);                    //check correct global value
    BOOST_CHECK_EQUAL(test.GetGlobalNy(), Ny);
    BOOST_CHECK_EQUAL(test.GetGlobalNpts(), Npts);

    BOOST_CHECK_CLOSE(test.GetGlobalLx(), Lx,tol);
    BOOST_CHECK_CLOSE(test.GetGlobalLy(), Ly,tol);
}

/**
 * @brief Test whether LidDrivenCavity::SetDomainSize assigns values correctly and correctly configures problem
 *************************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetDomainSize) {
    
    //default values in class, denoting global problem
    int Nx = 9;
    int Ny = 9;
    int Npts = 81;

    //values to assign and expectations -> these denote the global problem
    double Lx = 2.2;
    double Ly = 3.3;
    double dx = (double)Lx/(Nx-1);
    double dy = (double)Ly/(Ny-1);

    //set up MPI communicators and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy,iIgnore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    //default Nx, Ny is 9; should be split by constructor into localNx and localNy; Lx and Ly should also be split into correct localLx and localLy
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,iIgnore,iIgnore);

    LidDrivenCavity test;
    
    test.SetDomainSize(Lx,Ly);                                                  //call function to be tested, input global values
    
    //check whether local and global values changed correctly
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(test.GetLx(), localLx, tol);                              //check local values
    BOOST_CHECK_CLOSE(test.GetLy(), localLy, tol);
    BOOST_CHECK_CLOSE(test.GetDx(), dx, tol);
    BOOST_CHECK_CLOSE(test.GetDy(), dy, tol);
    BOOST_CHECK_EQUAL(test.GetNx(), localNx);
    BOOST_CHECK_EQUAL(test.GetNy(), localNy);
    BOOST_CHECK_EQUAL(test.GetNpts(), localNx*localNy);

    BOOST_CHECK_CLOSE(test.GetGlobalLx(), Lx, tol);                             //check global values
    BOOST_CHECK_CLOSE(test.GetGlobalLy(), Ly, tol);
    BOOST_CHECK_EQUAL(test.GetGlobalNx(), Nx);
    BOOST_CHECK_EQUAL(test.GetGlobalNy(), Ny);
    BOOST_CHECK_EQUAL(test.GetGlobalNpts(), Npts);
}

/**
 * @brief Test whether LidDrivenCavity::SetGridSize assigns values correctly and correctly configures problem
 ***********************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetGridSize) {
    
    //default values in class, denoting global problem
    double Lx = 1.0;
    double Ly = 1.0;

    //values to assign and expectations -> these denote the global problem
    int Nx = 102;
    int Ny = 307;                                                                   //grid points in x and y
    int Npts = Nx*Ny;
    double dx = (double)Lx/(Nx-1);
    double dy = (double)Ly/(Ny-1);

    //set up MPI communicators and split domain equally
    MPI_Comm grid,row,col;
    int localNx,localNy,iIgnore;
    double localLx,localLy;
    
    CreateCartGridVerify(grid,row,col);
    //default Nx, Ny is 9; should be split by constructor into localNx and localNy; Lx and Ly should also be split into correct localLx and localLy
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,localLx,localLy,iIgnore,iIgnore);
    
    LidDrivenCavity test;

    test.SetGridSize(Nx,Ny);                                                        //call function to be tested -> pass global value
    
    //check whether values changed correctly
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(test.GetLx(), localLx, tol);                                  //check local values
    BOOST_CHECK_CLOSE(test.GetLy(), localLy, tol);
    BOOST_CHECK_CLOSE(test.GetDx(), dx, tol);
    BOOST_CHECK_CLOSE(test.GetDy(), dy, tol);
    BOOST_CHECK_EQUAL(test.GetNx(), localNx);
    BOOST_CHECK_EQUAL(test.GetNy(),localNy);
    BOOST_CHECK_EQUAL(test.GetNpts(), localNx*localNy);

    BOOST_CHECK_CLOSE(test.GetGlobalLx(), Lx, tol);                                 //check global values
    BOOST_CHECK_CLOSE(test.GetGlobalLy(), Ly, tol);
    BOOST_CHECK_EQUAL(test.GetGlobalNx(), Nx);
    BOOST_CHECK_EQUAL(test.GetGlobalNy(), Ny);
    BOOST_CHECK_EQUAL(test.GetGlobalNpts(), Npts);
}

/**
 * @brief Test whether LidDrivenCavity::SetTimeStep assigns values correctly and correctly configures problem
 **********************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetTimeStep) {

    //variable to set, no need split domain as time step dt should be same across all processes
    double dt = 0.024;
    
    LidDrivenCavity test;

    test.SetTimeStep(dt);                                                           //call function to be tested
    
    //no other expected values, check dt
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(test.GetDt(), dt, tol);
}

/**
 * @brief Test whether LidDrivenCavity::SetFinalTime assigns values correctly and correctly configures problem
 ************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetFinalTime) {

    //variable to set, no need split domain as final time T should be same across all processes
    double T = 23.43;
    
    LidDrivenCavity test;

    test.SetFinalTime(T);                                                           //call function to be tested
    
    //no other expected values, check T only
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(T, test.GetT(), tol);
}

/**
 * @brief Test whether LidDrivenCavity::SetReynoldsNumber assigns values correctly and correctly configures problem
 ******************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SetReynoldsNumber) {
    //variable to set, no need split domain as Reynolds number Re should be same across all processes
    double Re = 5000;

    //expected values
    double U = 1.0;
    double nu = U/Re;
    
    LidDrivenCavity test;

    test.SetReynoldsNumber(Re);                                                 //call function to be tested
    
    //check whether changed values,  is actually changed correctly
    double tol = 1e-6;
    BOOST_CHECK_CLOSE(Re, test.GetRe(), tol);
    BOOST_CHECK_CLOSE(nu, test.GetNu(), tol);
    BOOST_CHECK_CLOSE(U, test.GetU(), tol);
}

/**
 * @brief Test case to confirm whether LidDrivenCavity::PrintConfiguration function prints out the correct global configuration
******************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_PrintConfiguration)
{
    /*Define a test case with different numbers for each variable to ensure each variable is printed correctly
    Other values for reference are flow speed U = 1.0, kinematic viscosity nu = 0.01 and step sizes dx = 0.05 and dy = 0.2
    This gives nu*dt/dx/dy = 0.2 < 0.25 so solver should not exit
    These describe the GLOBAL domain, not local
    */
    double dt   = 0.2;                                                      //time step
    double T    = 5.1;                                                      //final time
    int    Nx   = 21;                                                       //domain size in x and y direction
    int    Ny   = 11;
    double Lx   = 1.0;                                                      //domain length in x and y direction
    double Ly   = 2.0;
    double Re   = 100;                                                      //Reynolds number
    
    //expected string outputs, taking care that spacing is same as that from output, and formatting of numbers is same as cout
    string configGridSize = "Grid size: 21 x 11";
    string configSpacing = "Spacing:   0.05 x 0.2";
    string configLength = "Length:    1 x 2";
    string configGridPts = "Grid pts:  231";
    string configTimestep = "Timestep:  0.2";
    string configSteps = "Steps:     26";                                   //also test ability of ceiling, as 25.5 should round up to 26
    string configReynolds = "Reynolds number: 100";
    string configOther = "Linear solver: preconditioned conjugate gradient";

    //MPI implementation
    MPI_Comm grid,row,col;
    int rowRank,colRank;

    CreateCartGridVerify(grid,row,col);
    MPI_Comm_rank(row,&rowRank);
    MPI_Comm_rank(col,&colRank);                            //need to know as printing should only occur on root rank, or rowRank = 0 and colRank = 0

    LidDrivenCavity test;
    
    // Redirect cout to a stringstream for capturing the output
    std::stringstream terminalOutput;
    std::streambuf* printConfigData = std::cout.rdbuf(terminalOutput.rdbuf());

    // Invoke setting functions and print the solver configuration to stringstream
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    test.PrintConfiguration();

    // Restore the original cout
    std::cout.rdbuf(printConfigData);

    //check if desired string is present in the terminal output
    std::string output = terminalOutput.str();

    //root rank should print, so check data printed correctly
    if((rowRank == 0) & (colRank == 0)) {
        BOOST_CHECK(output.find(configGridSize) != std::string::npos);
        BOOST_CHECK(output.find(configSpacing) != std::string::npos);
        BOOST_CHECK(output.find(configLength) != std::string::npos);
        BOOST_CHECK(output.find(configGridPts) != std::string::npos);
        BOOST_CHECK(output.find(configTimestep) != std::string::npos);
        BOOST_CHECK(output.find(configSteps) != std::string::npos);
        BOOST_CHECK(output.find(configReynolds) != std::string::npos);
        BOOST_CHECK(output.find(configOther) != std::string::npos);
    }
    else {  //Other ranks should not have printed anything, so check that these strings do not exist

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
 * @brief Test whether LidDrivenCavity::Initialise initialises the vorticity, streamfunctions correctly
******************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_Initialise) {
    /*Define a test case with different numbers for each variable. These describe the GLOBAL domain, not local
    Other values for reference are flow speed U = 1.0, kinematic viscosity nu = 0.01 and step sizes dx = 0.05 and dy = 0.2
    This gives nu*dt/dx/dy = 0.2 < 0.25 so solver should not exit
    */
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;

   //split the domain and compute what each MPI communicator data should be for initialise
    MPI_Comm grid,row,col;
    int localNx,localNy,xStart,yStart;
    double dIgnore;

    CreateCartGridVerify(grid,row,col);
    SplitDomainMPIVerify(grid, Nx, Ny, Lx,Ly,localNx,localNy,dIgnore,dIgnore,xStart,yStart);    //compute local domain of each process
    int localNpts = localNx*localNy;                            //local number of points in process

    /*//compute ranks for adjacent grids for data transfer, if at boundary, returns -2 (MPI_PROC_NULL)
    int bottomRank,topRank;
    MPI_Cart_shift(col,0,1,&bottomRank,&topRank);//from bottom to top*/

    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    test.Initialise();                                          //initialise the problem
    
    double* v = new double[localNpts];
    double* s = new double[localNpts];
    
    test.GetData(v,s);                                          //get vorticity v and streamfunction s, expect to be filled with zeros
                                                                //note that GetDat returns local data stored in each process, not global

    //initialise implies zeros for all values, so norm v and norm s give the error norm
    double tol = 1e-6;
    double globalErrorV,globalErrorS;
    double eV = cblas_dnrm2(localNpts,v,1);                     
    double eS = cblas_dnrm2(localNpts,s,1);
    eV *= eV;                                                   //2-norm squared is linear and can be summed
    eS *= eS;                   

    MPI_Allreduce(&eV,&globalErrorV,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&eS,&globalErrorS,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    globalErrorV = sqrt(globalErrorV);                          //as Allreduce summed 2-norm squared, sqrt to get 2-norm
    globalErrorS = sqrt(globalErrorS);

    BOOST_CHECK_SMALL(globalErrorS, tol);                       //use SMALL not CLOSE as working with zeros
    BOOST_CHECK_SMALL(globalErrorV, tol);

    delete[] v;
    delete[] s;
}

/**
 * @brief Test whether LidDrivenCavity::WriteSolution() creates file and outputs correct data in correct format.
 * Uses initial condition data to check. Upon problem initialisation, streamfunction and voriticity should be zero everywhere. 
 * Vertical and horizontal velocities should be zero everywhere, except at top of lid where horizontal velocity is 1.
 *******************************************************************************************************************************/
BOOST_AUTO_TEST_CASE(LidDrivenCavity_WriteSolution) 
{
    /*Define a test case with different numbers for each variable. These describe the GLOBAL domain, not local
    Other values for reference are flow speed U = 1.0, kinematic viscosity nu = 0.01 and step sizes dx = 0.05 and dy = 0.2
    This gives nu*dt/dx/dy = 0.2 < 0.25 so solver should not exit
    */
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

    int localNx = Nx;                                                           //so that IDX can be used with value for global Nx
    
    //compute the expected values, note that vy, s and v should all be zero and vx should only be U for top boundary 
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
    MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);                       //get world rank

    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    test.Initialise();
    
    std::string fileName = "testOutput";                            //output initial conditions to file named testOutput
    test.WriteSolution(fileName);                                   //only produces one file, even in MPI

    std::ifstream outputFile(fileName);             //create stream for file, let each process read it as independent access for reading
    BOOST_REQUIRE(outputFile.is_open());            //check if file has been created by seeing if it can be opened, if doesn't exist, terminate
    
    /*read data from file and check initial conditions -> verifies WriteSolution();
    expect format of x,y,vorticity,streamfunction,vx,vy
    initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have vx = 1*/
    
    std::string line;                                               //variable to capture line of data from file
    std::string xData,yData,vData,sData,vxData,vyData;              //temporary string variables
    
    double* xDataSet = new double[Nx*Ny];
    double* yDataSet = new double[Nx*Ny];
    double* vDataSet = new double[Nx*Ny];
    double* sDataSet = new double[Nx*Ny];
    double* vxDataSet = new double[Nx*Ny];
    double* vyDataSet = new double[Nx*Ny];

    int i = 0;                                                      //denote Cartesian coordinates (i,j)
    int j = 0;
    int dataPoints = 0;                                             //counter to track number of data points printed, should equal Nx*Ny
    unsigned int dataCol = 0;                                       //conter for number of columns per row, should be 6

    while(std::getline(outputFile,line)) {                          //keep getting line if line exists
        
        if(line.empty()) {
            continue;                                               //if empty line in file, skip as it separates the data points
        }
        
        //first check that there are six datapoints per row
        std::stringstream dataCheck(line); 
        
        while (dataCheck >> xData) {
                dataCol++;                                          //calculate how many data points per line
        }
        
        BOOST_REQUIRE(dataCol == 6);                                //check data printed correctly, 6 data points per row
        dataCol = 0;                                                //reset data counter
        
        //now read data into correct array in correct location
        std::stringstream data(line);
        data >> xData >> yData >> vData >> sData >> vxData >> vyData;

        xDataSet[IDX(i,j)] = std::stod(xData);
        yDataSet[IDX(i,j)] = std::stod(yData);
        vDataSet[IDX(i,j)] = std::stod(vData);
        sDataSet[IDX(i,j)] = std::stod(sData);
        vxDataSet[IDX(i,j)] = std::stod(vxData);
        vyDataSet[IDX(i,j)] = std::stod(vyData);
        
        dataPoints++;                                               //log one more data point
        
        //also increment i and j counters; note that data is written out column by column, so i constant, j increments
        j++;
        if(j >= Ny) {       //if j exceeds number of rows (or Ny), then we want index to point to the start of next column 
            j = 0;
            i++;
        }
    }
    outputFile.close();

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
}

/**
 * @brief Tests whether the time domain solver LidDrivenCavity::Integrator works correctly by comparing problem to a reference dataset
 * @note Reference dataset generated via serial version of this solver
 *******************************************************************************************************************************/
/*BOOST_AUTO_TEST_CASE(LidDrivenCavity_Integrator) 
{
    //take a case where steady state is reached -> rule of thumb, fluid should pass through at least 10 times to reach SS
    //For reference dx dy = 0.005 and nu*dt/dx/dy = 0.2 < 0.25
    double dt   = 0.005;
    double T    = 20;
    int    Nx   = 201;
    int    Ny   = 201;
    double Lx   = 1;
    double Ly   = 1;
    double Re   = 1000;

    int localNx = Nx;                                           //so that IDX can be used with global value of Nx
    
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

    //output file containing initial condition, in file name IntegratorTest; reference data in DataIntegratorTestCase
    std::string fileName = "IntegratorTest";
    std::string refData = "DataIntegratorTestCase";
    test.WriteSolution(fileName);
    
    std::ifstream outputFile(fileName);             // Create streams for outputted data and reference data
    std::ifstream refFile(refData);
    
    BOOST_REQUIRE(outputFile.is_open());            //check if output file has been created by seeing if it can be opened
    BOOST_REQUIRE(refFile.is_open());               //check if reference data file exists by seeing if it can be opened

    //read data from file and compare with reference data from serial solver
    //expect format of x,y,vorticity,streamfunction,vx,vy
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
    outputFile.close();
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
}*/