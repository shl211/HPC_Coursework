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

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Macro to map matrix entry i,j onto it's corresponding location in memory, assuming column-wise matrix storage
 * @param I     matrix index i denoting the ith row
 * @param J     matrix index j denoting the jth columns
 */
#define IDX(I,J) ((J)*Nx + (I))                     //define a new operation to improve computation?

/**
 * @brief Test SolverCG::Solve where if input b is very close to zero, then output x should be exactly 0.0 for all entries
 */
BOOST_AUTO_TEST_CASE(SolverCG_NearZeroInput)
{
    const int Nx = 10;                                      //define grid and steps
    const int Ny = 10;
    double dx = 0.1;
    double dy = 0.1;    
    int n = Nx*Ny;                                          //total number of grid points
    
    double *b = new double[n];                              //allocate memory of input b and output x, denotes equation Ax = b
    double *x = new double[n];
    
    SolverCG test(Nx,Ny,dx,dy);                             //create test solver

    for(int i = 0; i < n; i++) {
        b[i] = 1e-8;                                        //100 element array with each element = 1e-8
    }                                                       //2-norm of b is smaller than tol*tol where tol = 1e-3 as specified in SolverCG 
    
    test.Solve(b,x);                                        //Solve Ax=b for x
    
    for(int i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(x[i],0.0);                        //check all terms of x are exactly 0.0
    }                                                       //use equal instead of close for double as 0.0 should be written into x
    
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
    const int Nx = 100;                                 //define number of grids with correct step sizes
    const int Ny = 100;
    double dx = (double)Lx/(Nx - 1);
    double dy = (double)Ly/(Ny - 1);    
    int n = Nx*Ny;
    double tol = 1e-3;                                  //tolerance as specified in SolverCG
    double *b = new double[n];                          //allocate memory
    double *x = new double[n];
    double* x_actual = new double[n];

    SolverCG test(Nx,Ny,dx,dy);                         //create test solver
    
    std::srand(time(0));
    for(int i = 0; i < n; i++) {
        b[i] = 0.0;                                     //initialise b
        x[i] = (double) rand()/RAND_MAX;                //randomise x in range of [0,1]
    }
    
    //impose zero BC on edges
    for (int i = 0; i < Nx; ++i) {
        x[IDX(i, 0)] = 0.0;                             //zero BC on bottom surface
        x[IDX(i, Ny-1)] = 0.0;                          //zero BC on top surface
    }

    for (int j = 0; j < Ny; ++j) {
        x[IDX(0, j)] = 0.0;                             //zero BC on left surface
        x[IDX(Nx - 1, j)] = 0.0;                        //zero BC on right surface
    }

    for (int i = 0; i < Nx; ++i) {                      //generate the sinusoidal test case input b
        for (int j = 0; j < Ny; ++j) {
            b[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    
    test.Solve(b,x);                                    //Solve the sinusoidal test case
    
    for(int i = 0; i < Nx; ++i){                        //Generate the analytical solution x
        for(int j = 0; j < Ny; ++j) {
            x_actual[IDX(i,j)] = - sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
        }
    }

    for(int i = 0; i < n; ++i) {
        BOOST_CHECK_SMALL(x_actual[i]-x[i],tol);         //check error for each term between analytical and solver x is within tolerance
    }

    delete[] x;                                          //deallocate memory
    delete[] x_actual;
    delete[] b;
}

//test setting functions as well as print config, unable to test separate case of print config where nu > 0.25 as program terminates
/**
 * @brief Test case to confirm setting functions work for configuring the solver. Tests this via the PrintConfiguration function.
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_SettingConfiguration)
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

    LidDrivenCavity test;                           //default constructor
    
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

    BOOST_REQUIRE(output.find(configGridSize) != std::string::npos);
    BOOST_REQUIRE(output.find(configSpacing) != std::string::npos);
    BOOST_REQUIRE(output.find(configLength) != std::string::npos);
    BOOST_REQUIRE(output.find(configGridPts) != std::string::npos);
    BOOST_REQUIRE(output.find(configTimestep) != std::string::npos);
    BOOST_REQUIRE(output.find(configSteps) != std::string::npos);
    BOOST_REQUIRE(output.find(configReynolds) != std::string::npos);
    BOOST_REQUIRE(output.find(configOther) != std::string::npos);
}

/**
 * @brief Test whether problem has been initialised correctly and whether WriteSolution() creates file and outputs correct data in correct format
 * Upon problem initialisation, streamfunction and voriticity should be zero everywhere. Vertical and horizontal velocities should be zero 
 * everywhere, except at top of lid where horizontal velocity is 1.
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_InitialiseFileOutput) 
{
    //take previous working test case, same variable definitions as before
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;
    double dx = 0.05;
    double dy = 0.2;
    
    //set up lid driven cavity class and configure the problem
    LidDrivenCavity test;
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    test.Initialise();                              //initialise the problem
    
    std::string fileName = "testOutput";            //output initial conditions to file named testOutput
    test.WriteSolution(fileName);
    
    std::ifstream outputFile(fileName);             //create stream for file
    BOOST_REQUIRE(outputFile.is_open());            //check if file has been created by seeing if it can be opened, if doesn't exist, terminate
    
    //read data from file and check initial conditions -> verifies Initialise() and WriteSolution();
    //expect format of (x,y), (vorticity, streamfunction), (vx,vy)
    //initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have vx = 1
    
    std::string line;                                               //variable to capture line of data from file
    std::string xData,yData,vData,sData,vxData,vyData;              //temporary string variables
    double x,y,v,s,vx,vy;                                           //temporary double variables
    
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
        
        x = std::stod(xData);
        y = std::stod(yData);
        v = std::stod(vData);
        s = std::stod(sData);
        vx = std::stod(vxData);
        vy = std::stod(vyData);
        
        BOOST_CHECK_CLOSE(x,i*dx,1e-6);        //cehck x and y data printed in correct format, column by column
        BOOST_CHECK_CLOSE(y,j*dy,1e-6);
        
        BOOST_CHECK_CLOSE(v,0.0,1e-6);         //vorticity and streamfunction should be zero everywhere
        BOOST_CHECK_CLOSE(s,0.0,1e-6);
                                                //check velocity x and y are zeros; if top surface, then vx = 1
        if( std::abs(y - Ly) < 1e-6) {          //check if top surface via tolerance as it is double format, tolerance must be << dy
            BOOST_CHECK_CLOSE(vx,1.0,1e-6);     
            BOOST_CHECK_CLOSE(vy,0.0,1e-6);
        }
        else {
            BOOST_CHECK_CLOSE(vx,0.0,1e-6);
            BOOST_CHECK_CLOSE(vy,0.0,1e-6);
        }
        
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
    
    //delete file from directory to clean up
    if(std::remove(fileName.c_str()) == 0) {
        std::cout << fileName << " successfully deleted" << std::endl;
    }
    else {
        std::cout << fileName << " could not be deleted" << std::endl;
    }
}

/**
 * @brief Tests whether the time domain solver Integrator() works correctly by computing 5 time steps for a specified problem.
 * Solution is compared to the output generated by the serial solver, whose data is stored in a text file named "DataIntegratorTestCase".
 */
BOOST_AUTO_TEST_CASE(LidDrivenCavity_Integrator) 
{
    //take previous working test case, same representations as before
    double dt   = 0.01;
    double T    = 0.05;
    int    Nx   = 101;
    int    Ny   = 101;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 1000;
    //double dx = 0.01;                     //for reference, step sizes
    //double dy = 0.01;
    //nu * dt/dx/dy = 0.1 < 0.25            //for reference, check the step size limit is smaller than 0.25
    
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
    //expect format of (x,y), (vorticity, streamfunction), (vx,vy)
    //initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have vx = 1
    
    std::string line,refLine;                                   //variable to capture line of data from file
    std::string xData,yData,vData,sData,vxData,vyData;          //temporary string variables
    double x,y,v,s,vx,vy;                                       //temporary double variables
    double xRef,yRef,vRef,sRef,vxRef,vyRef;
    
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
        
        x = std::stod(xData);
        y = std::stod(yData);
        v = std::stod(vData);
        s = std::stod(sData);
        vx = std::stod(vxData);
        vy = std::stod(vyData);
        
        dataRef >> xData >> yData >> vData >> sData >> vxData >> vyData;
        
        xRef = std::stod(xData);
        yRef = std::stod(yData);
        vRef = std::stod(vData);
        sRef = std::stod(sData);
        vxRef = std::stod(vxData);
        vyRef = std::stod(vyData);
        
        //check data with reference values for this problem, generated by the serial solver
        BOOST_CHECK_CLOSE(x,xRef,1e-6);
        BOOST_CHECK_CLOSE(y,yRef,1e-6);
        
        BOOST_CHECK_CLOSE(v,vRef,1e-6);
        BOOST_CHECK_CLOSE(s,sRef,1e-6);
        
        BOOST_CHECK_CLOSE(vx,vxRef,1e-6);
        BOOST_CHECK_CLOSE(vy,vyRef,1e-6);
        
        dataPoints++;                           //log extra data pionts
    }
    outputFile.close();                         //close files
    refFile.close();
    
    BOOST_CHECK(dataPoints == Nx*Ny);            //check right number of grid point data is outputted
    
    //delete output file from directory to clean up
    if(std::remove(fileName.c_str()) == 0) {
        std::cout << fileName << " successfully deleted" << std::endl;
    }
    else {
        std::cout << fileName << " could not be deleted" << std::endl;
    }
}
