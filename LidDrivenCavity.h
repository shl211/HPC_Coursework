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
     */
    LidDrivenCavity();
    
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
     * @brief Get the total number of grid points in x direction, for testing purposes
     * @return Total number of grid points in x direction
     */
    int GetGlobalNx();

    /**
     * @brief Get the total number of grid points in y direction, for testing purposes
     * @return Total number of grid points in y direction
     */
    int GetNy();
    
    /**
     * @brief Get the total number of grid points in y direction, for testing purposes
     * @return Total number of grid points in y direction
     */
    
    int GetGlobalNy();
    /**
     * @brief Get total number of grid points, for testing purposes
     * @return Total number of grid points
     */
    int GetNpts();
    
    /**
     * @brief Get total number of grid points, for testing purposes
     * @return Total number of grid points
     */
    int GetGlobalNpts();

    /**
     * @brief Get domain length in x direction, for testing purposes
     * @return Domain length in x direction
     */
     double GetLx();

         /**
     * @brief Get domain length in x direction, for testing purposes
     * @return Domain length in x direction
     */
     double GetGlobalLx();
    
    /**
     * @brief Get domain length in y direction, for testing purposes
     * @return Domain length in y direction
     */
     double GetLy();
    
        /**
     * @brief Get domain length in y direction, for testing purposes
     * @return Domain length in y direction
     */
     double GetGlobalLy();
    
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
     */
    void GetData(double* vOut, double* sOut);


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
    double* vNext = nullptr;                ///<Pointer to array describing vorticity at next time step
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

    MPI_Comm comm_Cart_grid;
    MPI_Comm comm_row_grid;                 ///<MPI communicator for the process row in Cartesian topology grid
    MPI_Comm comm_col_grid;                 ///<MPI communicator for the process column in Cartesian topology grid
    int size;                               ///<Size of a row/column communicator, where size*size is the total number of processors
    int globalNx;                               ///<global Nx
    int globalNy;                               ///<global Ny
    double globalLx;
    double globalLy;

    int rowRank;///<rank of current process in comm_row_grid
    int colRank;///<rank of current process in comm_col_grid
    int topRank;///<rank of process above current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing above
    int leftRank;///<rank of process to left of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to left
    int rightRank;///<rank of process to right of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to right
    int bottomRank;///<rank of process to bottom of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing below

    bool boundaryDomain; ///<denotes whether the process is at the boundary of the CCartesian grid

    MPI_Request dataToLeft;
    MPI_Request dataToRight;
    MPI_Request dataToUp;
    MPI_Request dataToDown;

    double* vTopData = nullptr;              ///<Buffer to store the data 1 row above top of local grid
    double* vLeftData = nullptr;             ///<Buffer to store the data 1 column to left of local grid
    double* vRightData = nullptr;             ///<Buffer to store the data 1 column to right of local grid
    double* vBottomData = nullptr;              ///<Buffer to store the data 1 row below bottom of local grid
    double* sTopData = nullptr;              ///<Buffer to store the data 1 row above top of local grid
    double* sLeftData = nullptr;             ///<Buffer to store the data 1 column to left of local grid
    double* sRightData = nullptr;             ///<Buffer to store the data 1 column to right of local grid
    double* sBottomData = nullptr;              ///<Buffer to store the data 1 row below bottom of local grid
    
    double* tempLeft;
    double* tempRight;

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

    /**
   * @brief Setup Cartesian grid and column and row communicators
   * @param[out] comm_Cart_Grid   Communicator for Cartesian grid
   * @param[out] comm_row_grid    Communicator for current row of Cartesian grid
   * @param[out] comm_col_grid    Communicator for current column of Cartesian grid
   * @param[out] size     size of communicators
   */
    void CreateCartGrid(MPI_Comm &cartGrid,MPI_Comm &rowGrid, MPI_Comm &colGrid);

    /**
     * @brief Split the global grid size into local grid size based off MPI grid size
     * @param[in] grid      MPI Cartesian grid
     * @param[in] globalNx  Global Nx domain to be discretised
     * @param[in] globalNy Global Ny domain to be discretised
     * @param[in] globalLx  Global Lxx
     * @param[in] globalLy
     * @param[out] localNx  Domain size Nx for each local process
     * @param[out] localNy  Domain size Ny for each local process
     * @param[out] localLx
     * @param[out] localLy
     */
    void SplitDomainMPI(MPI_Comm &grid, int globalNx, int globalNy, double globalLx, double globalLy, int &localNx, int &localNy, double &localLx, double &localLy) ;
};

