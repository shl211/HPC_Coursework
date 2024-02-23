default: LidDrivenCavitySolver

#create object files
LidDrivenCavitySolver.o: LidDrivenCavitySolver.cpp LidDrivenCavity.h
	g++ -std=c++11 -Wall -o2 -o LidDrivenCavitySolver.o -c LidDrivenCavitySolver.cpp

LidDrivenCavity.o: LidDrivenCavity.cpp LidDrivenCavity.h SolverCG.h 
	g++ -std=c++11 -Wall -o2 -o LidDrivenCavity.o -c LidDrivenCavity.cpp

SolverCG.o: SolverCG.cpp SolverCG.h
	g++ -std=c++11 -Wall -o2 -o SolverCG.o -c SolverCG.cpp

#link
LidDrivenCavitySolver: LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
	g++ -o solver LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o -lboost_program_options -lblas

#clean
.PHONY: clean

clean:
	-rm -f *.o solver