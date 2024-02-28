CXX = g++
CXXFLAGS = -std=c++11 -Wall -o2
TARGET = solver
OBJS = LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
HDRS = LidDrivenCavity.h SolverCG.h
LDLIBS = -lboost_program_options -lblas
DOXYFILE = Doxyfile
TESTTARGET = unittests
TESTS = unittest.cpp

default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

all: $(TARGET)

doc:
	doxygen Doxyfile

$(TESTTARGET):
	$(CXX) $(CXXFLAGS) -o $@ unittest.cpp SolverCG.cpp $(LDLIBS)

.PHONY: clean

clean:
	-rm -f *.o $(TARGET) $(TESTTARGET)
