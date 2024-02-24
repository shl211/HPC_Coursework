#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>

#define IDX(I,J) ((J)*Nx + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Default constructor
 */ 
LidDrivenCavity::LidDrivenCavity()
{
}

/**
 * @brief Destructor to deallocate memory allocated on the heap
 */
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/**
 * @brief Specify the domain size, recomputes grid spacing dx and dy
 * @param xlen  Length of domain in the x direction
 * @param ylen  Length of domain in the y direction
 */
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();                   //update grid spacing dx dy based off new domain
}

/**
 * @brief Specify the grid size, recomputes grid spacing dx and dy
 * @param nx    Number of grid points in the x direction
 * @param ny    Number of grid points in the y direction
 */
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();                   //update grid spacing dx dy based off new domain
}

/**
 * @brief Specify the time step
 * @param deltat    Time step
 */
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/**
 * @brief Specify the final time for solver
 * @param finalt    Final time in seconds
 */
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/**
 * @brief Specify the Reynolds number
 * @param re    Reynolds number
 */
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

/**
 * @brief Set up solver by allocating memory and creating spatial solver of class SolverCG
 */
void LidDrivenCavity::Initialise()
{
    CleanUp();

    v   = new double[Npts]();
    s   = new double[Npts]();
    tmp = new double[Npts]();
    cg  = new SolverCG(Nx, Ny, dx, dy);
}

/**
 * @brief Execute the time domain solver, which implicitly calls the spatial domain solver at each time step
 */ 
void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);                            //number of time steps required
    for (int t = 0; t < NSteps; ++t)
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl;                         //after each step, output time and step information
        Advance();                                      //?
    }
}

/**
 * @brief Print the solution to a specified file
 * @param file      file name
 */ 
void LidDrivenCavity::WriteSolution(std::string file)
{
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;             //compute velocity in x direction at every grid point from streamfunction
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;             //compute velocity in y direction at every grid point from streamfunction
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;                                                //impose x velocity as U at top surface for no-slip boundary condition
    }

    std::ofstream f(file.c_str());                                          //open file for output
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)                                        //print data in columns (i.e. along y direction, keep x direction constant)
        {
            k = IDX(i, j);                                                  //index in array
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k]     //on each line in file, print the grid location (x,y), vorticity...
              << " " << u0[k] << " " << u1[k] << std::endl;                 //streamfunction, x velocity, y velocity at that grid location
        }
        f << std::endl;                                                     //After printing all y data for certain x, proceed to next x column with a space...
    }                                                                       //to differentiate between each x column grid
    f.close();//close file

    delete[] u0;//deallocate memory
    delete[] u1;
}

/**
 * @brief print to terminal the current problem configuration
 */
void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

/**
 * @brief Deallocate memory
 */
void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] s;
        delete[] tmp;
        delete cg;
    }
}

/**
 * @brief Updates spatial steps dx and dy based on current Nx,Ny,Lx,Ly
 */
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx * Ny;
}

/**
 * @brief Computes vorticity and streamfunction for next time step
 */
void LidDrivenCavity::Advance()
{
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;

    // Boundary node vorticity
    for (int i = 1; i < Nx-1; ++i) {
        // top
        v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
        // bottom
        v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)])
                       - 2.0 * dxi*U;
    }
    for (int j = 1; j < Ny-1; ++j) {
        // left
        v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);
        // right
        v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);
    }

    // Compute interior vorticity
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            v[IDX(i,j)] = dx2i*(
                    2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);
        }
    }

    // Time advance vorticity
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            v[IDX(i,j)] = v[IDX(i,j)] + dt*(
                ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
                 *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
                 *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
              + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
        }
    }

    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver
    /*
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    */

    // Solve Poisson problem
    cg->Solve(v, s);
}
