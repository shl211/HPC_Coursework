#include <iostream>
#include <cmath>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <mpi.h>
#include "LidDrivenCavity.h"

/**
 * @brief Main program that allows for user specification of problem followed by implementation of time and spatial solvers
 */
int main(int argc, char* argv[])
{
    //-----------------------------------------Initialise MPI communicator-----------------------------------------//
    int worldRank, size, retval_rank, retval_size;    
    MPI_Init(&argc, &argv);
    
    //return rank and size
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); 
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //check if communicator set up correctly
    if(retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {
        cout << "Invalid communicator" << endl;
        return 1;
    }
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));   //round sqrt to nearest whole number
    
    if((p*p != size) | (size < 1)) {                   //if not a square number, print error and terminate program
        if(worldRank == 0)                                       //print only on root rank
            cout << "Invalide process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        return 1;
    }
    
    //set up Cartesian topology to represent the 'grid' nature of the problem
    MPI_Comm comm_Cart_Grid;
    const int dims = 2;                                 //2 dimensions in grid
    int gridSize[dims] = {p,p};                         //p processes per dimension
    int periods[dims] = {0,0};                          //grid is not periodic
    int reorder = 1;                                    //reordering of grid
    MPI_Cart_create(MPI_COMM_WORLD,dims,gridSize,periods,reorder, &comm_Cart_Grid);        //create Cartesian topology
    
    //extract coordinates
    MPI_Comm comm_row_grid, comm_col_grid;                          //communicators for rows and columns of grid
    int gridRank;
    int coords[dims];
    int keep[dims];
    
    retval_rank = MPI_Comm_rank(comm_Cart_Grid, &gridRank);         //retrieve rank in grid, also check if grid created successfully
    if(retval_rank == MPI_ERR_COMM) {
        if (worldRank == 0)
            cout << "Cartesian grid was not created" << endl;
            
        MPI_Finalize();
        return 1;
    }
    
    MPI_Cart_coords(comm_Cart_Grid, gridRank, dims, coords);        //generate coordinates
    
    keep[0] = 0;        //create row communnicator in subgrid, process can communicate with other processes on row
    keep[1] = 1;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_row_grid);
    
    keep[0] = 1;        //create column communnicator in subgrid, process can communicate with other processes on column
    keep[1] = 0;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_col_grid);
    
    //cout << "Rank " << gridRank+1 << " has coords (" << coords[0] << "," << coords[1] << ")" << endl;
    
    //------------------------------------User program options to define problem ------------------------------------//
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;                                                       //extract user inputs
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {       
        if(worldRank == 0)                                                           //only print on global root rank
            cout << opts << endl;
        
        MPI_Finalize();
        return 0;
    }
    
    //------------------------------After user inputs, divide domain appropraitely across the grid --------------------------//
    //divide the domain size into as even of chunks/grids as possible, following the Cartesian grid
    //so each process has different size Lx and Ly, and differente size Nx and Ny, but  same everything else
    int xDomainSize,yDomainSize;                        //local domain sizes in each direction, or local Nx and Ny
    int rowStart,rowEnd,colStart,colEnd;                //denotes index of where in problem domain the process accesses directly
    double xDomainLength,yDomainLength;                    //local lengths of Nx and Ny
    int rem;
    
    xDomainSize = vm["Nx"].as<int>() / p;           //minimum size of each process in x and y domain
    yDomainSize = vm["Ny"].as<int>() / p;
    
    //first assign for x dimension
    rem = vm["Nx"].as<int>() % p;                   //remainder, denotes how many processes need an extra row in domain
    
    if(coords[0] < rem) {//safer to use coordinates (row) than rank, which could be reordered, if coord(row)< remainder, use minimum + 1
        yDomainSize++;
        rowStart = yDomainSize * coords[0];             //index denoting starting row in local domain
        rowEnd = rowStart + yDomainSize;                //index denoting final row in local domain
    }
    else {//otherwise use minimum, and find other values
        rowStart = (yDomainSize + 1) * rem + yDomainSize * (coords[0] - rem);           //starting row accounts for previous processes with +1 rows and +0 rows
        rowEnd = rowStart + yDomainSize;
    }
    
    //same for y dimension
    rem = vm["Ny"].as<int>() % p;
        
    if(coords[1] < rem) {//safer to use coordinates (column) than rank, which could be reordered, if coord(column)< remainder, use minimum + 1
        xDomainSize++;
        colStart = xDomainSize * coords[1];             //index denoting starting column in local domain
        colEnd = colStart + xDomainSize;                //index denoting final column in local domain
    }
    else {//otherwise use minimum, and find other values
        colStart = (xDomainSize + 1) * rem + xDomainSize * (coords[1] - rem);           //starting column accounts for previous processes with +1 rows and +0 rows
        colEnd = colStart + xDomainSize;
    }
    
    xDomainLength = vm["Lx"].as<double>() * xDomainSize/vm["Nx"].as<int>();            //calculate new local domain lengths, this ensures dx and dy same as serial case
    yDomainLength = vm["Ly"].as<double>() * yDomainSize/vm["Ny"].as<int>();

    //-----------------------------------------------------Parallel Solver---------------------------------------------------//

    LidDrivenCavity* solver = new LidDrivenCavity(comm_row_grid,comm_col_grid,coords[0],coords[1]);
                                                                                //define solver and specify problem with user inputs
    solver->SetDomainSize(xDomainLength,yDomainLength);                         //define each local solver domain
    solver->SetGridSize(xDomainSize,yDomainSize);
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());

    solver->PrintConfiguration();                                               //print the solver configuration to user

    solver->Initialise();                                                       //initialise solver

    solver->WriteSolution("ic.txt");                                            //write initial state to file named ic.txt

    solver->Integrate();                                                        //perform time integration, implicitly calls spatial domain solver

    solver->WriteSolution("final.txt");                                         //write the final solution to file named final.txt

    MPI_Finalize();
	return 0;
}
