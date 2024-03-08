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
    int rank, size, retval_rank, retval_size;    
    MPI_Init(&argc, &argv);
    
    //return rank and size
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //check if communicator set up correctly
    if(retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {
        cout << "Invalid communicator" << endl;
        return 1;
    }
    
    //check if input rank is square number size = p^2
    double sqrtSize = sqrt(size);
    sqrtSize = round(sqrtSize);     //round sqrt to nearest whole number
    
    if(((int) sqrtSize*sqrtSize != size) | (size < 1)) {                   //if not a square number, print error and terminate program
        if(rank == 0)                                       //print only on root rank
            cout << "Invalide process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        return 1;
    }
    
    //set up Cartesian topology to represent the 'grid' nature of the problem
    MPI_Comm comm_Cart_Grid;
    const int dims = 2;                                 //2 dimensions in grid
    int gridSize[dims] = {(int)sqrtSize,(int)sqrtSize};     //p processes per dimension
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
        if (rank == 0)
            cout << "Cartesian grid was not created" << endl;
            
        MPI_Finalize();
        return 1;
    }
    
    MPI_Cart_coords(comm_Cart_Grid, gridRank, dims, coords);
    
    keep[0] = 0;        //create row communnicator in subgrid
    keep[1] = 1;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_row_grid);
    
    keep[0] = 1;        //create column communnicator in subgrid
    keep[1] = 0;
    MPI_Cart_sub(comm_Cart_Grid, keep, &comm_col_grid);
    
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
        if(rank == 0)                                                       //only print on root rank
            cout << opts << endl;
        
        MPI_Finalize();
        return 0;
    }

    //-----------------------------------------------------Parallel Solver---------------------------------------------------//

    LidDrivenCavity* solver = new LidDrivenCavity();                            //define solver and specify problem with user inputs
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
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
