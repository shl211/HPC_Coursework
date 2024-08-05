#include <iostream>
#include <cmath>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <mpi.h>
#include "LidDrivenCavity.h"

/**
 * @brief Main program that allows for user specification of problem followed by implementation of solver
 * @warning MPI ranks must satisfy \f$ P = p^2 \f$, otherwise program will terminate
 *********************************************************************************************************************/
int main(int argc, char* argv[])
{
    //-----------------------------------------Initialise MPI communicator-----------------------------------------//
    int worldRank, size, retval_rank, retval_size;    
    MPI_Init(&argc, &argv);
    
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);                                //return rank and size
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &size);
   
    if(retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {                        //check if communicator set up correctly
        cout << "Invalid communicator" << endl;
        return 1;
    }
    
    //check if input rank is square number size = p^2
    int p = round(sqrt(size));
    
    if((p*p != size) | (size < 1)) {                                                        //if not a square number, print error and terminate program
        if(worldRank == 0)
            cout << "Invalide process size. Process size must be square number of size p^2 and greater than 0" << endl;
            
        MPI_Finalize();
        return 2;
    }
    
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

    //extract user inputs
    po::variables_map vm;                                                       
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {       
        if(worldRank == 0)
            cout << opts << endl;
        
        MPI_Finalize();
        return 0;
    }

    //don't let user use excessive number of processes for the specified grid size
    //for example no point using 4x4 processes to compute anything smaller than 8x8 grid, would be slower
    //protect code from a small bug in processing certain special cases that occur for above case, leads to slightly erroneous solution
    //also catches case where a process ends up having no data to process
    if((vm["Nx"].as<int>()*2 < p) || (vm["Ny"].as<int>()*2 < p)) {

        if(worldRank == 0)
            cout << "Excessive number of processes (p^2) for specified grid size. Ensure 2*Nx < p and 2*Ny < p" << endl;

        MPI_Finalize();
        return 3;
    }

    //------------------------------------------Implement Parallel Solver---------------------------------------------------//
    //pass global values in, LidDrivenCavity will perform suitable domain discretistion
    //this allows the Set variables to retain their 'global' meaning, so user not confused by 'local' and 'global' domain definitions

    LidDrivenCavity* solver = new LidDrivenCavity();

    solver->SetDomainSize(vm["Lx"].as<double>(),vm["Ly"].as<double>());         //configure the problem with user inputs
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());

    solver->PrintConfiguration();                                               //print the solver configuration to user

    solver->Initialise();                                                       //initialise solver

    solver->WriteSolution("ic.txt");                                            //write initial state to file named ic.txt

    solver->Integrate();                                                        //solve the flow properties at each time step and grid point

    solver->WriteSolution("final.txt");                                         //write the final solution to file named final.txt

    MPI_Finalize();
	return 0;
}
