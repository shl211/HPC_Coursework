CXX = g++
CXXFLAGS = -std=c++11 -Wall -o2
TARGET = solver
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
HDRS = LidDrivenCavity.h SolverCG.h
LDLIBS = -lboost_program_options -lblas
DOXYFILE = Doxyfile

default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

all: $(TARGET)

.PHONY: clean doc

clean:
	-rm -f *.o solver

doc:
	doxygen Doxyfile