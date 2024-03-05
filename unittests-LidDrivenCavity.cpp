#include <iostream>
#include <sstream>
#include <string>
#include <streambuf>
#include <cmath>
#include <cstdio>

#include "LidDrivenCavity.h"
#include "SolverCG.h"

#define BOOST_TEST_MODULE LidDrivenCavity
#include <boost/test/included/unit_test.hpp>

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
