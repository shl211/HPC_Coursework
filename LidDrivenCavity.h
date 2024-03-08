#pragma once

#include <string>
using namespace std;

class SolverCG;

/**
 * @class LidDrivenCavity
 * @brief Class that describes the properties of the lid driven cavity problem. The fluid flow in this problem can be characterised 
in both time and space (x,y). This class contains methods that allow for the 2D incompressible Navier-Stokes equations to be evaluated
on the problem domain \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$, where \f$ L_x \f$ is the domain length in x direction and \f$ L_y \f$ is the
domain length in the y direction. The problem time domain is \f$ t\in[0,T_f] \f$ where \f$ T_f \f$ is the final time.
Results can be outputted to a text file.
 */
class LidDrivenCavity
{
public:
    /**
     * @brief Constructor containing information on MPI process within a Cartesian topology
     * @param[in] rowGrid   MPI communicator for the process row in Cartesian topology grid
     * @param[in] colGrid   MPI communicator for the process column in Cartesian topology grid
     * @param[in] rowRank   Denotes the row the MPI process is within a Cartesian topology grid i.e.coordinates[0]
     * @param[in] colRank   Denotes the column the MPI process is within a Cartesian topology grid i.e.coordinates[1]
     */
    LidDrivenCavity(MPI_Comm &rowGrid, MPI_Comm &colGrid, int rowRank, int colRank);
    
    /**
     * @brief Destructor to deallocate memory
     */
    ~LidDrivenCavity();
    
    /**
     * @brief Get the time step dt, for testing purposes
     * @return Time step dt
     */
    double GetDt(); 
    
    /**
     * @brief Get the final time T, for testing purposes
     * @return Final time T
     */
    double GetT();
    
    /**
     * @brief Get the x direction step size dx, for testing purposes
     * @return x direction step size dx
     */
    double GetDx();
    
     /**
     * @brief Get the y direction step size dy, for testing purposes
     * @return y direction step size dy
     */
    double GetDy();
    
    /**
     * @brief Get the total number of grid points in x direction, for testing purposes
     * @return Total number of grid points in x direction
     */
    int GetNx();
    
    /**
     * @brief Get the total number of grid points in y direction, for testing purposes
     * @return Total number of grid points in y direction
     */
    int GetNy();
    
    /**
     * @brief Get total number of grid points, for testing purposes
     * @return Total number of grid points
     */
    int GetNpts();
    
    /**
     * @brief Get domain length in x direction, for testing purposes
     * @return Domain length in x direction
     */
     double GetLx();
    
    /**
     * @brief Get domain length in y direction, for testing purposes
     * @return Domain length in y direction
     */
     double GetLy();
    
    /**
     * @brief Get Reynolds number, for testing purposes
     * @return Reynolds number
     */
    double GetRe();
     
     /**
      * @brief Get horizontal flow velocity at top lid, for testing purposes
      * @return Horizontal flow vevlocity at top lid
      */
    double GetU();
      
      /**
       * @brief Get kinematic viscosity, for testing purposes
       * @return Kinematic viscosity
       */
    double GetNu();

    /**
     * @brief Get vorticity, streamfunction, and velocity data. Results written into the provided pointers.
     * @param[out] vOut    pointer to array containing vorticity
     * @param[out] sOut    pointer to array containing streamfunction
     * @param[out] u0Out   pointer to array containing velocity in x direction
     * @param[out] u1Out   pointer to array containing velocity in y direction
     */
    void GetData(double* vOut, double* sOut, double* u0Out, double* u1Out);


    /**
     * @brief Specify the problem domain size \f$ (x,y)\in[0,xlen]\times[0,ylen] \f$ and recomputes grid spacing dx and dy
     * @param[in] xlen  Length of domain in the x direction
     * @param[in] ylen  Length of domain in the y direction
     */
    void SetDomainSize(double xlen, double ylen);
    
    /**
     * @brief Specify the grid size \f$ N_x \times N_y \f$ and recomputes grid spacing dx and dy
     * @param[in] nx    Number of grid points in the x direction
     * @param[in] ny    Number of grid points in the y direction
     */
    void SetGridSize(int nx, int ny);
    
    /**
     * @brief Specify the time step for the solver
     * @param[in] deltat    Time step
     */
    void SetTimeStep(double deltat);
    
    /**
     * @brief Specify the final time for solver
     * @param[in] finalt    Final time
     */
    void SetFinalTime(double finalt);
    
    /**
     * @brief Specify the Reynolds number of the flow at the top of the lid
     * @param[in] re    Reynolds number
     */
    void SetReynoldsNumber(double Re);

    /**
     * @brief Initialise solver by allocating memory and creating the initial condition, with vorticity and streamfunction zero everywhere.
     * The spatial solver of class SolverCG is also created.
     */
    void Initialise();
    
    /**
     * @brief Execute the time domain solver from 0 to T in steps of dt. Calls the spatial domain solver at each time step. Also displays 
     * progress of the solver.
     */ 
    void Integrate();
    
    /**
     * @brief Print grid position (x,y), voriticity, streamfunction and velocities (vx,vy) to a text file with the specified name. 
     * If the specified file does not exist, then it will create a text file with the specified name and output data there.
     * @param[in] file      name of the target text file
     */ 
    void WriteSolution(std::string file);
    
    /**
     * @brief Print to terminal the current problem specification
     */
    void PrintConfiguration();

private:
    double* v   = nullptr;                  ///<Pointer to array describing vorticity
    double* s   = nullptr;                  ///<Pointer to array describing streamfunction
    double* tmp = nullptr;                  ///<Temporary array

    double dt   = 0.01;                     ///<Time step for solver, default 0.01
    double T    = 1.0;                      ///<Final time for solver, default 1
    double dx;                              ///<Grid spacing in x direction
    double dy;                              ///<Grid spacing in y direction
    int    Nx   = 9;                        ///<Number of grid points in x direction, default 9
    int    Ny   = 9;                        ///<Number of grid points in y direction, default 9
    int    Npts = 81;                       ///<Total number of grid points, default 81
    double Lx   = 1.0;                      ///<Length of domain in x direction, default 1
    double Ly   = 1.0;                      ///<Length of domain in y direction, default 1
    double Re   = 10;                       ///<Reynolds number, default 10
    double U    = 1.0;                      ///<Horizontal velocity at top of lid, default 1
    double nu   = 0.1;                      ///<Kinematic viscosity, default 0.1

    MPI_Comm comm_row_grid;                 ///<MPI communicator for the process row in Cartesian topology grid
    MPI_Comm comm_col_grid;                 ///<MPI communicator for the process column in Cartesian topology grid
    int MPIcoords[2];                        ///<Coordinate of MPI process in a Cartesian topology grid
    int size;                               ///<Size of a row/column communicator, where size*size is the total number of processors
    
    SolverCG* cg = nullptr;                 ///<conjugate gradient solver for Ax=b that can solve spatial domain aspect of the problem

    /**
     * @brief Deallocate memory associated with arrays and classes
     */
    void CleanUp();
    
    /**
     * @brief Updates spatial steps dx and dy based on current grid point numbers (Nx,Ny) and domain lengths (Lx,Ly)
     */
    void UpdateDxDy();
    
    /**
     * @brief Computes vorticity and streamfunction for each grid point in the problem for the next time step
     */
    void Advance();
};

