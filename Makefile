CXX = mpicxx -fopenmp
CXXFLAGS = -std=c++11 -Wall -o2 -g
TARGET = solver
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
HDRS = LidDrivenCavity.h SolverCG.h
LDLIBS = -lboost_program_options -lblas
TESTTARGET = unittests
TESTS = unittests.cpp LidDrivenCavity.cpp SolverCG.cpp
OTHER = testOutput IntegratorTest ic.txt final.txt html latex	#other files/directories that should be deleted

default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

all: $(TARGET)

doc:
	doxygen Doxyfile

$(TESTTARGET): $(TESTS) 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: clean

clean:
	-rm -rf *.o $(TARGET) $(TESTTARGET) $(OTHER)
