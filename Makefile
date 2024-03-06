CXX = g++
CXXFLAGS = -std=c++11 -Wall -o2 -ftree-vectorize
TARGET = solver
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
HDRS = LidDrivenCavity.h SolverCG.h
LDLIBS = -lboost_program_options -lblas
DOXYFILE = Doxyfile
TESTTARGET = unittests
TESTS = unittests.cpp LidDrivenCavity.cpp SolverCG.cpp

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
	-rm -f *.o $(TARGET) $(TESTTARGET)
