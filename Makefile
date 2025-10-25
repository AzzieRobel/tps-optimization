# High-Performance GPT-OSS-20B Inference Engine Makefile
# Optimized for x86_64 with AVX2/AVX-512 support

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -mavx2 -mfma -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl
LDFLAGS = -pthread -lm

# Debug flags (uncomment for debugging)
# CXXFLAGS += -g -DDEBUG -fsanitize=address -fsanitize=undefined

# Release flags for maximum performance
CXXFLAGS += -DNDEBUG -flto -fno-exceptions -fno-rtti
LDFLAGS += -flto

# Source files
SRCDIR = src
SOURCES = $(SRCDIR)/inference_engine.cpp $(SRCDIR)/main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = inference_engine

# Include directories
INCLUDES = -I$(SRCDIR)

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete! Target: >30 TPS"

# Compile source files
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete!"

# Install dependencies (if needed)
install-deps:
	@echo "Installing system dependencies..."
	@echo "This project has no external dependencies!"
	@echo "All optimizations are implemented in pure C++"

# Run performance benchmark
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	./$(TARGET) benchmark "The future of artificial intelligence"

# Run interactive mode
interactive: $(TARGET)
	@echo "Starting interactive mode..."
	./$(TARGET) interactive

# Run memory test
memory-test: $(TARGET)
	@echo "Running memory optimization test..."
	./$(TARGET) benchmark "Memory optimization test"

# Generate sample text
generate: $(TARGET)
	@echo "Generating sample text..."
	./$(TARGET) generate "The future of artificial intelligence"

# Performance profiling
profile: $(TARGET)
	@echo "Running performance profiling..."
	perf record -g ./$(TARGET) benchmark "Performance profiling test"
	perf report

# Memory profiling
memcheck: $(TARGET)
	@echo "Running memory check..."
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./$(TARGET) benchmark "Memory check test"

# CPU optimization flags
cpu-optimize:
	@echo "CPU optimization flags:"
	@echo "  -march=native: Use native CPU instructions"
	@echo "  -mavx2: Enable AVX2 vector instructions"
	@echo "  -mavx512f: Enable AVX-512 foundation instructions"
	@echo "  -mavx512dq: Enable AVX-512 doubleword and quadword instructions"
	@echo "  -mavx512cd: Enable AVX-512 conflict detection instructions"
	@echo "  -mavx512bw: Enable AVX-512 byte and word instructions"
	@echo "  -mavx512vl: Enable AVX-512 vector length extensions"
	@echo "  -mfma: Enable fused multiply-add instructions"

# Show build information
info:
	@echo "Build Information:"
	@echo "  Compiler: $(CXX)"
	@echo "  Flags: $(CXXFLAGS)"
	@echo "  Sources: $(SOURCES)"
	@echo "  Target: $(TARGET)"
	@echo "  Threads: $(shell nproc)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the inference engine (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  benchmark    - Run performance benchmark"
	@echo "  interactive  - Start interactive mode"
	@echo "  generate     - Generate sample text"
	@echo "  profile      - Run performance profiling"
	@echo "  memcheck     - Run memory check"
	@echo "  cpu-optimize - Show CPU optimization flags"
	@echo "  info         - Show build information"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all clean install-deps benchmark interactive memory-test generate profile memcheck cpu-optimize info help

