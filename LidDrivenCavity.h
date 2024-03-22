#pragma once

#include <string>
using namespace std;

class SolverCG;

/**
 * @class LidDrivenCavity
 * @brief Class that describes the properties of the lid driven cavity problem.
 * 
 * <table>
 *   <tr>
 *     <td>
 *       @image html domain.png "Lid driven cavity domain"
 *     </td>
 *     <td>
 *       @image html discreteDomain.png "Lid driven cavity discretised domain"
 *     </td>
 *   </tr>
 * </table>
 * 
 * The fluid flow in this problem can be characterised in both time and space \f$ (x,y) \f$. This class contains methods that allow for the 
 * 2D incompressible Navier-Stokes equations to be evaluated via streamfunctions and vorticity on the problem domain 
 * \f$ (x,y)\in[0,L_x]\times[0,L_y] \f$, where \f$ L_x \f$ is the domain length in \f$ x \f$ direction and \f$ L_y \f$ 
 * is the domain length in the \f$ y \f$ direction. The problem time domain is \f$ t\in[0,T_f] \f$ 
 * where \f$ T_f \f$ is the final time. Results can be outputted to a text file.
 * 
 * @note When implemented with MPI, LidDrivenCavity expects inputs to be describe the undiscretised global domain as 
 * discretisation is done within this solver. This is to prevent the user from being confused over whether the Set functions
 * should take local or global values. However, all private member variables store local values not global values, unless
 * otherwise stated.
 * 
 * @note Row major storage format is used for matrices
 * 
 * @warning MPI ranks must satisfy \f$ P = p^2 \f$, otherwise program will terminate
 ***********************************************************************************************************************************************/
class LidDrivenCavity
{
public:
    /**
     * @brief Constructor that sets up the MPI implementation of this class
     *******************************************************************************************************************************************/
    LidDrivenCavity();
    
    /**
     * @brief Destructor to deallocate memory
     ********************************************************************************************************************************************/
    ~LidDrivenCavity();

   /**
     * @defgroup GetLDC Get LidDrivenCavity Local Domain Parameters
     * 
     * Return a value that describes the local lid driven cavity problem domain stored in a process, for testing purposes
     * @return The value that describes an aspect of the local lid driven cavity problem domain
     * @note Returned values describe the local domain of a process, which are not necessarily the same as the global domain (unless MPI ranks = 1)
     * @{
     ****************************************************************************************************************************************/
    int GetNx();                        ///<Get the total number of local grid points in x direction
    int GetNy();                        ///<Get the total number of local grid points in y direction
    int GetNpts();                      ///<Get the total number of grid points in local domain
    double GetLx();                     ///<Get local domain length in x direction
    double GetLy();                     ///<Get local domain length in y direction
    /**@}*/
    
    /**
     * @defgroup GetGLDC Get LidDrivenCavity Global Domain Parameters
     * 
     * Return a value that describes the global lid driven cavity problem domain, for testing purposes
     * @return The value that describes an aspect of the lid driven cavity problem domain
     * @note The returned global values will be same across all processes. The member functions without Global in the name are parameters
    that are the same regardless of the problem discretisation. 
     * @{
    *********************************************************************************************************************************************/
    int GetGlobalNx();                  ///<Get the total number of global grid points in x direction 
    int GetGlobalNy();                  ///<Get the total number of global grid points in y direction
    int GetGlobalNpts();                ///<Get the total number of grid points in global domain
    double GetGlobalLx();               ///<Get global domain length in x direction
    double GetGlobalLy();               ///<Get global domain length in y direction
    double GetRe();                     ///<Get Reynolds number
    double GetU();                      ///<Get horizontal flow velocity at top lid
    double GetNu();                     ///<Get kinematic viscosity
    double GetDt();                     ///<Get the time step dt
    double GetT();                      ///<Get the final time T
    double GetDx();                     ///<Get the x direction step size dx
    double GetDy();                     ///<Get the y direction step size dy
    /**@}*/

    /**
     * @brief Get local vorticity and streamfunction
     * @note It is assumed that the user will provide the correct array sizes 
     * @param[out] vOut    Vorticity at all grid points
     * @param[out] sOut    Streamfunction at all grid points
     ************************************************************************************************************************************************/
    void GetData(double* vOut, double* sOut);

    /**
     * @brief Specify the problem domain size \f$ (x,y)\in[0,xlen]\times[0,ylen] \f$ and recomputes grid spacing \f$ dx \f$ and \f$ dy \f$
     * @note This takes in values for the global domain
     * @param[in] xlen  Length of global domain in the x direction
     * @param[in] ylen  Length of global domain in the y direction
     */
    void SetDomainSize(double xlen, double ylen);
    
    /**
     * @brief Specify the grid size \f$ N_x \times N_y \f$ and recomputes grid spacing \f$ dx \f$ and \f$ dy \f$
     * @note This takes in values for the global domain
     * @param[in] nx    Number of grid points in the x direction in global domain
     * @param[in] ny    Number of grid points in the y direction in global domain
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
     * @brief Initialise solver
     * 
     * Solver initialised by allocating memory and creating the initial condition, with vorticity and streamfunction zero everywhere.
     * The spatial solver of class SolverCG is also created.
     */
    void Initialise();
    
    /**
     * @brief Compute the flow field for the lid driven cavity at a specified time.
     * 
     * Execute the time domain solver from 0 to T in steps of dt. Calls the spatial domain solver at each time step. Also displays progress of the solver.
     */ 
    void Integrate();
    
    /**
     * @brief Print grid position \f$ (x,y) \f$, voriticity, streamfunction and velocities to a text file with the specified name. 
     * 
     * If the specified file does not exist, then it will create a text file with the specified name and output data there.
     * @param[in] file      name of the target text file
     */ 
    void WriteSolution(std::string file);
    
    /**
     * @brief Print to terminal the current problem specification
     */
    void PrintConfiguration();

private:
    double* v   = nullptr;                  ///<Vorticity at current time step
    double* vNext = nullptr;                ///<Vorticity at new time step
    double* s   = nullptr;                  ///<Pointer to array describing streamfunction
    double* tmp = nullptr;                  ///<Temporary array

    double dt   = 0.01;                     ///<Time step for solver, default 0.01
    double T    = 1.0;                      ///<Final time for solver, default 1
    double dx;                              ///<Grid spacing in x direction
    double dy;                              ///<Grid spacing in y direction
    int    Nx   = 9;                        ///<Number of local grid points in x direction, default 9
    int    Ny   = 9;                        ///<Number of local grid points in y direction, default 9
    int    Npts = 81;                       ///<Total number of local grid points, default 81
    double Lx   = 1.0;                      ///<Length of local domain in x direction, default 1
    double Ly   = 1.0;                      ///<Length of local domain in y direction, default 1
    double Re   = 10;                       ///<Reynolds number, default 10
    double U    = 1.0;                      ///<Horizontal velocity at top of lid, default 1
    double nu   = 0.1;                      ///<Kinematic viscosity, default 0.1

    MPI_Comm comm_Cart_grid;                ///<MPI communicator describing a Cartesian topology grid
    MPI_Comm comm_row_grid;                 ///<MPI communicator for the process row in #comm_Cart_grid
    MPI_Comm comm_col_grid;                 ///<MPI communicator for the process column in #comm_Cart_grid
    int size;                               ///<Size of a row/column communicator, where size*size is the total number of MPI processes
    int globalNx;                           ///<Number of global grid points in x direction
    int globalNy;                           ///<Number of global grid points in y direction
    double globalLx;                        ///<Length of global domain in x direction
    double globalLy;                        ///<Length of global domain in y direction
    int xDomainStart;                       ///<For the x direction, denotes where the local domain starts in the context of the global domain 
    int yDomainStart;                       ///<For the y direction, denotes where the local domain starts in the context of the global domain 

    int rowRank;                            ///<Rank of current process in #comm_row_grid
    int colRank;                            ///<Rank of current process in #comm_col_grid
    int topRank;                            ///<Rank of process above current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing above
    int bottomRank;                         ///<Rank of process to bottom of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing below
    int leftRank;                           ///<Rank of process to left of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to left
    int rightRank;                          ///<Rank of process to right of current process in Cartesian grid, -2 (MPI_PROC_NULL) if nothing to right

    bool boundaryDomain;                    ///<Denotes whether the process is at the boundary of the Cartesian grid #comm_Cart_grid

    /// MPI_Request handle to check data send -> [0] = send to top, [1] = send to bottom, [2] = send left, [3] = send right
    MPI_Request requests[4];

    double* vTopData = nullptr;             ///<Buffer to store the vorticity data 1 row above top of local grid
    double* vBottomData = nullptr;          ///<Buffer to store the vorticity data 1 row below bottom of local grid
    double* vLeftData = nullptr;            ///<Buffer to store the vorticity data 1 column to left of local grid
    double* vRightData = nullptr;           ///<Buffer to store the vorticity data 1 column to right of local grid
    double* sTopData = nullptr;             ///<Buffer to store the streamfunction data 1 row above top of local grid
    double* sBottomData = nullptr;          ///<Buffer to store the streamfunction data 1 row below bottom of local grid
    double* sLeftData = nullptr;            ///<Buffer to store the streamfunction data 1 column to left of local grid
    double* sRightData = nullptr;           ///<Buffer to store the streamfunction data 1 column to right of local grid
    
    double* tempLeft;                       ///<Temporarily stores data for left hand side of current local grid, to be sent left
    double* tempRight;                      ///<Temporarily stores data for right hand side of current local grid, to be sent right

    SolverCG* cg = nullptr;                 ///<Conjugate gradient solver for Ax=b that can solve spatial domain aspect of the problem

    /**
     * @brief Deallocate memory associated with arrays and classes
     *****************************************************************************************************************************************/
    void CleanUp();
    
    /**
     * @brief Updates spatial steps #dx and #dy based on current grid point numbers (#Nx,#Ny) and domain lengths (#Lx,#Ly)
     ******************************************************************************************************************************************/
    void UpdateDxDy();
    
    /**
     * @brief Computes vorticity and streamfunction for each grid point in the problem for the next time step
     ******************************************************************************************************************************************/
    void Advance();

    /**
     * @brief Computes vorticity at the current time step from streamfunction at the current time step
     * 
     * @bug Currently small bug with processing special cases where local domain is single cell, column vector (\f$ N_x \times 1 \f$) or row vector
     * (\f$ 1 \times N_y \f$). This error will lead to an erroneous solution. However, not a big issue, as realistically this case should never be
     * encountered, as it is a very inefficient use of resources. For example, why would anyone solve a 5x5 grid with 16 processors? 
     * This is only an issue if very small problems are used relative to number of processors, and until this bug is fixed, LidDrivenCavitySolver.cpp
     * will prevent user from entering values that can lead to issues.
     ******************************************************************************************************************************************/
    void ComputeVorticity();

    /**
     * @brief Computes time advanced vorticity from the vorticity and streamfunction at the current time step
     ******************************************************************************************************************************************/
    void ComputeTimeAdvanceVorticity();

    /**
     * @brief Compute the velocity at all grid points from the streamfunction
     * @param[out] u0   Horizontal velocity
     * @param[out] u1   Vertical velocity
     ******************************************************************************************************************************************/
    void ComputeVelocity(double* u0, double* u1);

    /**
   * @brief Setup Cartesian grid and column and row communicators
   * @param[out] cartGrid   Communicator for Cartesian grid
   * @param[out] rowGrid    Communicator for current row of Cartesian grid
   * @param[out] colGrid    Communicator for current column of Cartesian grid
   *********************************************************************************************************************************************/
    void CreateCartGrid(MPI_Comm &cartGrid,MPI_Comm &rowGrid, MPI_Comm &colGrid);

    /**
     * @brief Split the global grid size into local grid size based off MPI grid size
     * @param[in] grid      MPI Cartesian grid
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
     */
    void SplitDomainMPI(MPI_Comm &grid, int globalNx, int globalNy, double globalLx, double globalLy,
                     int &localNx, int &localNy, double &localLx, double &localLy, int &xStart, int &yStart);
};

