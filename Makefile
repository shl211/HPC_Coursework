# Compiler and flags
CXX = mpicxx -fopenmp
CXXFLAGS = -std=c++11 -Wall -O2
LDLIBS = -lboost_program_options -lblas

# Targets and sources
TARGET = solver
OBJS = src/LidDrivenCavitySolver.o src/LidDrivenCavity.o src/SolverCG.o
HDRS = include/LidDrivenCavity.h include/SolverCG.h
TESTTARGET = unittests
TESTOBJS = test/unittests.o src/LidDrivenCavity.o src/SolverCG.o

# Other files/directories that should be deleted
OTHER = testOutput IntegratorTest ic.txt final.txt html latex

# Default target
default: $(TARGET)

# Pattern rule for object files in src directory
src/%.o: src/%.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ -c $<

# Pattern rule for object files in test directory
test/%.o: test/%.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ -c $<

# Build the main target
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

# Build all targets
all: $(TARGET)

# Generate documentation
doc:
	doxygen Doxyfile

# Build the test target
$(TESTTARGET): $(TESTOBJS)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $^ $(LDLIBS)

# Clean up generated files
.PHONY: clean

clean:
	-rm -rf src/*.o test/*.o $(TARGET) $(TESTTARGET) $(OTHER)
