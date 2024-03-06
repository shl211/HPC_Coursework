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

#define IDX(I,J) ((J)*Nx + (I)) //define a new operation

//test solver, implicity tests constructor and all other private functions
//for case where b is practically 0
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


//for a sinusoidal case with zero BCs and random guess for x
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
        x[i] = (double) rand()/RAND_MAX;
    }
    
    //impose zero BC on edges
    for (int i = 0; i < Nx; ++i) {
        x[IDX(i, 0)] = 0.0;             //zero BC on bottom surface
        x[IDX(i, Ny-1)] = 0.0;          //zero BC on top surface
    }

    for (int j = 0; j < Ny; ++j) {
        x[IDX(0, j)] = 0.0;             //zero BC on left surface
        x[IDX(Nx - 1, j)] = 0.0;        //zero BC on right surface
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
        //BOOST_REQUIRE(abs(x[i] - x_actual[i]) < 1e-3);
        //std::cout << x[i] << std::endl;
        BOOST_CHECK_SMALL(x[i]-x_actual[i],1e-3);
    }

    delete[] x;
    delete[] x_actual;
    delete[] b;
}

//test setting functions as well as print config, unable to test separate case of print config where nu > 0.25 as program terminates
BOOST_AUTO_TEST_CASE(Setting)
{
    //test case
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    //int    Npts = 231;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;
    //double U    = 1.0;
    //double nu   = 0.01;
    //double dx = 0.05;
    //double dy = 0.2;

    //note that nu * dt / dx / dy = 0.2 < 0.25
    
    //expected string outputs, taking care that spacing is same as that from output, and formatting of numbers is same as cout
    string configGridSize = "Grid size: 21 x 11";
    string configSpacing = "Spacing:   0.05 x 0.2";
    string configLength = "Length:    1 x 2";
    string configGridPts = "Grid pts:  231";
    string configTimestep = "Timestep:  0.2";
    string configSteps = "Steps:     26";           //also test ability of ceiling, as 25.5 should round up to 26
    string configReynolds = "Reynolds number: 100";
    string configOther = "Linear solver: preconditioned conjugate gradient";

    LidDrivenCavity test;//default constructor
    
    // Redirect cout to a stringstream for capturing the output
    std::stringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf());

    // Call the PrintConfiguration function
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    test.PrintConfiguration();

    // Restore the original cout
    std::cout.rdbuf(oldCout);

    // Check the output in the stringstream
    std::string output = buffer.str();

    // Perform Boost Test assertions on the expected output
    BOOST_REQUIRE(output.find(configGridSize) != std::string::npos);
    BOOST_REQUIRE(output.find(configSpacing) != std::string::npos);
    BOOST_REQUIRE(output.find(configLength) != std::string::npos);
    BOOST_REQUIRE(output.find(configGridPts) != std::string::npos);
    BOOST_REQUIRE(output.find(configTimestep) != std::string::npos);
    BOOST_REQUIRE(output.find(configSteps) != std::string::npos);
    BOOST_REQUIRE(output.find(configReynolds) != std::string::npos);
    BOOST_REQUIRE(output.find(configOther) != std::string::npos);
}

//test solver and file output together
BOOST_AUTO_TEST_CASE(FileOutput) 
{
    //take previous working test case
    double dt   = 0.2;
    double T    = 5.1;
    int    Nx   = 21;
    int    Ny   = 11;
    double Lx   = 1.0;
    double Ly   = 2.0;
    double Re   = 100;
    double dx = 0.05;
    double dy = 0.2;
    
    //set up lid driven cavity class
    LidDrivenCavity test;//default constructor
    test.SetDomainSize(Lx,Ly);
    test.SetGridSize(Nx,Ny);
    test.SetTimeStep(dt);
    test.SetFinalTime(T);
    test.SetReynoldsNumber(Re);
    
    //initialise and write the output
    test.Initialise();
    
    //output file containing initial condition, in file name testOutput
    std::string fileName= "testOutput";
    test.WriteSolution(fileName);
    
     // Create an output file stream
    std::ifstream outputFile(fileName);
    BOOST_REQUIRE(outputFile.is_open());            //check if file has been created by seeing if it can be opened
    
    //read data from file and check initial conditions -> verifies Initialise() and WriteSolution();
    //expect format of (x,y), (vorticity, streamfunction), (vx,vy)
    //initial condition, so vorticity and streamfunction should be zeros everywhere, and only top surface should have ux = 1
    std::string line;
    std::string xData,yData,vData,sData,vxData,vyData;
    double x,y,v,s,vx,vy;
    
    int i = 0;
    int j = 0;
    int dataPoints = 0;            //counter to track number of data points printed, should equal Nx*Ny
    unsigned int dataCol = 0;               //conter for number of columns per row, should be 6
    //while file is open
    while(std::getline(outputFile,line)) {
        
        if(line.empty()) {
            continue;               //if empty line, skip
        }
        
        //first check that there are six
        std::stringstream dataCheck(line);
        
        while (dataCheck >> xData) {
                dataCol++;
        }
        BOOST_REQUIRE(dataCol == 6);                    //check data printed correctly, 6 data points per row
        dataCol = 0;   //reset counter
        
        //now read data into correct place
        std::stringstream data(line);
        data >> xData >> yData >> vData >> sData >> vxData >> vyData;
        
        x = std::stod(xData);
        y = std::stod(yData);
        v = std::stod(vData);
        s = std::stod(sData);
        vx = std::stod(vxData);
        vy = std::stod(vyData);
        
        //cehck x and y data printed in correct format, column by column
        BOOST_CHECK_CLOSE(x,i*dx,1e-6);
        BOOST_CHECK_CLOSE(y,j*dy,1e-6);
        
        //vorticity and streamfunction should be zero everywhere
        BOOST_CHECK_CLOSE(v,0.0,1e-6);
        BOOST_CHECK_CLOSE(s,0.0,1e-6);
        
        //check velocity x and y are zeros; if top surface, then vx = 1
        if( std::abs(y - Ly) < 1e-6) {          //check if top surface, doubles, so check with tolerance
            BOOST_CHECK_CLOSE(vx,1.0,1e-6);
            BOOST_CHECK_CLOSE(vy,0.0,1e-6);
        }
        else {
            BOOST_CHECK_CLOSE(vx,0.0,1e-6);
            BOOST_CHECK_CLOSE(vy,0.0,1e-6);
        }
        
        dataPoints++;//increment data points
        
        //also increment i and j counters; note that data is written out column by column, so i constant, j increments
        j++;
        if(j >= Ny) {       //if j exceeds number of rows (or Ny), then we want index to point to the start of next column 
            j = 0;
            i++;
        }
    }
    outputFile.close();
    
    BOOST_CHECK(dataPoints == Nx*Ny);            //check right number of grid point data is outputted
    
    //delete file from directory to clean up#
    if(std::remove(fileName.c_str()) == 0) {
        std::cout << fileName << " successfully deleted" << std::endl;
    }
    else {
        std::cout << fileName << " could not be deleted" << std::endl;
    }
}