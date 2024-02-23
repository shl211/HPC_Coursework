CC = g++
CXXFLAGS = -std=c++11 -Wall -o2
LDLIBS = -lboost_program_options -lblas
HDRS = LidDrivenCavity.h SolverCG.h

default: LidDrivenCavitySolver

LidDrivenCavitySolver.o: LidDrivenCavitySolver.cpp $(HDRS)

LidDrivenCavity.o: LidDrivenCavity.cpp $(HDRS)

SolverCG.o: SolverCG.cpp SolverCG.h $(HDRS)

LidDrivenCavitySolver: LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
	g++ -o solver LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o $(LDLIBS)

.PHONY: clean

clean:
	-rm -f *.o solver