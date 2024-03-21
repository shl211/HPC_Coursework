# HPC Lid Driven Cavity Coursework 2024

Parallel implementation of a lid driven cavity fluid solver with MPI and OpenMP. Written by Stephen Liu.

## Table of Contents

- [Overview](#Overview)
- [Running](#Running-the-Code)



## Overview

As part of the HPC coursework, a serial lid driven cavity solver is parallelised with MPI and OpenMP. The problem to be solved is seen in Figure 1. The flow properties in the cavity at any time t is desired and must be computed via the 2D Navier-Stokes equation, which can be solved via discretised streamfunction and vorticity. A grid discretisation, in the form of a second-order central differencing equation, is generated, as seen in Figure 2. A preconditioned conjugate solver is used to solve the spatial aspect of the 2D Navier-Stokes equation. The time-domain aspect of the problem is then also solved by a five point stencil. By sequentially solving the spatial and time problem, the flow properties at any time t can be computed.

![Lid Driven Cavity Domain](domain.png)

Figure 1

![Discrete Lid Driven Cavity](discreteDomain.png)
Figure 2

## Running the Code
