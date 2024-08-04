# Compiler and flags
CXX = mpicxx -fopenmp
CXXFLAGS = -std=c++11 -Wall -O2
LDLIBS = -lboost_program_options -lblas

# Directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objects
BIN_DIR = $(BUILD_DIR)/executables

# Targets and sources
TARGET = solver
OBJS = $(OBJ_DIR)/LidDrivenCavitySolver.o $(OBJ_DIR)/LidDrivenCavity.o $(OBJ_DIR)/SolverCG.o
HDRS = include/LidDrivenCavity.h include/SolverCG.h
TESTTARGET = unittests
TESTOBJS = $(OBJ_DIR)/unittests.o $(OBJ_DIR)/LidDrivenCavity.o $(OBJ_DIR)/SolverCG.o

# Other files/directories that should be deleted
OTHER = testOutput IntegratorTest ic.txt final.txt docs/html docs/latex

# Default target
default: $(TARGET)

# Pattern rule for object files in src directory
$(OBJ_DIR)/%.o: src/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ -c $<

# Pattern rule for object files in test directory
$(OBJ_DIR)/%.o: test/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ -c $<

# Build the main target
$(BIN_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDLIBS)
	@ln -sf $@ $(TARGET)

# Build the test target
$(BIN_DIR)/$(TESTTARGET): $(TESTOBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $^ $(LDLIBS)
	@ln -sf $@ $(TESTTARGET)

# Convenience targets for default target names
$(TARGET): $(BIN_DIR)/$(TARGET)
$(TESTTARGET): $(BIN_DIR)/$(TESTTARGET)

# Build all targets
all: $(TARGET) $(TESTTARGET)

# Generate documentation
doc:
	doxygen docs/Doxyfile

# Clean up generated files
.PHONY: clean

clean:
	-rm -rf $(BUILD_DIR) $(TARGET) $(TESTTARGET) $(OTHER)
