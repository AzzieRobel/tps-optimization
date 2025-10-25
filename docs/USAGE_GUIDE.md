# Usage Guide

## Quick Start

### Building the Project

```bash
# Clone and build
git clone <repository-url>
cd TPS
make clean && make -j$(nproc)

# Verify build
./inference_engine --help
```

### Basic Usage

```bash
# Generate text with default settings
./inference_engine generate "The future of artificial intelligence"

# Run performance benchmark
./inference_engine benchmark "Performance test prompt"

# Interactive mode
./inference_engine interactive
```

## Command Line Interface

### Available Commands

#### 1. Generate Text
```bash
./inference_engine generate <prompt> [options]
```

**Options:**
- `--max-tokens <N>`: Maximum tokens to generate (default: 512)
- `--temperature <T>`: Sampling temperature (default: 0.7)
- `--threads <N>`: Number of threads (default: auto-detect)

**Example:**
```bash
./inference_engine generate "Explain quantum computing" --max-tokens 200 --temperature 0.5
```

#### 2. Performance Benchmark
```bash
./inference_engine benchmark [prompt] [options]
```

**Options:**
- `--runs <N>`: Number of benchmark runs (default: 10)
- `--tokens <N>`: Tokens per run (default: 100)
- `--threads <N>`: Number of threads (default: auto-detect)

**Example:**
```bash
./inference_engine benchmark "The future of AI" --runs 20 --tokens 200
```

#### 3. Interactive Mode
```bash
./inference_engine interactive [options]
```

**Options:**
- `--max-tokens <N>`: Maximum tokens per generation (default: 100)
- `--temperature <T>`: Sampling temperature (default: 0.7)

**Example:**
```bash
./inference_engine interactive --max-tokens 150 --temperature 0.8
```

## Configuration

### Environment Variables

```bash
# Set number of threads
export INFERENCE_THREADS=16

# Set cache size (in MB)
export INFERENCE_CACHE_SIZE=1024

# Enable debug mode
export INFERENCE_DEBUG=1

# Set quantization level (0=FP16, 1=INT8, 2=INT4)
export INFERENCE_QUANTIZATION=1
```

### Configuration File

Create `config.json`:
```json
{
    "model": {
        "path": "gpt-oss-20b.bin",
        "quantization": "int8",
        "cache_size": 1073741824
    },
    "inference": {
        "max_tokens": 512,
        "temperature": 0.7,
        "threads": 0
    },
    "optimization": {
        "simd": true,
        "quantization": true,
        "caching": true
    }
}
```

## API Usage

### C++ API

```cpp
#include "inference_engine.h"

// Initialize engine
InferenceEngine engine;
engine.load_model("gpt-oss-20b.bin");
engine.initialize_quantization();

// Configure
engine.set_num_threads(16);
engine.set_cache_size(1024 * 1024 * 1024);
engine.enable_quantization(true);

// Generate text
std::vector<Token> prompt = {1, 2, 3, 4, 5}; // Your tokenized prompt
auto result = engine.generate(prompt, 100, 0.7f);

// Get performance stats
auto stats = engine.get_stats();
std::cout << "TPS: " << stats.get_tps() << std::endl;
```

### Batch Processing

```cpp
// Process multiple prompts
std::vector<std::vector<Token>> prompts = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

auto results = engine.generate_batch(prompts, 100, 0.7f);

for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "Result " << i << ": ";
    for (auto token : results[i]) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
}
```

## Performance Tuning

### Hardware-Specific Optimization

#### Intel CPUs
```bash
# Enable AVX-512
export CXXFLAGS="-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl"

# NUMA optimization
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

#### AMD CPUs
```bash
# Enable AVX2
export CXXFLAGS="-mavx2 -mfma"

# Memory optimization
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
```

### Memory Optimization

```cpp
// Set optimal cache sizes
engine.set_l1_cache_size(1024 * 1024);        // 1MB L1 cache
engine.set_l2_cache_size(1024 * 1024 * 1024); // 1GB L2 cache

// Enable memory mapping
engine.enable_memory_mapping(true);

// Set page size
engine.set_page_size(4096);
```

### Quantization Tuning

```cpp
// Dynamic quantization
engine.enable_dynamic_quantization(true);

// Set quantization level
engine.set_quantization_level(1); // 0=FP16, 1=INT8, 2=INT4

// Enable adaptive quantization
engine.enable_adaptive_quantization(true);
```

## Monitoring and Debugging

### Performance Monitoring

```bash
# Real-time performance monitoring
watch -n 1 'ps aux | grep inference_engine'

# Memory usage monitoring
watch -n 1 'free -h'

# CPU utilization
htop
```

### Profiling

```bash
# Performance profiling
perf record -g ./inference_engine benchmark
perf report

# Memory profiling
valgrind --tool=memcheck --leak-check=full ./inference_engine benchmark

# Cache analysis
perf stat -e cache-misses,cache-references ./inference_engine benchmark
```

### Debug Mode

```bash
# Enable debug output
export INFERENCE_DEBUG=1
./inference_engine generate "Test prompt"

# Verbose logging
export INFERENCE_VERBOSE=1
./inference_engine benchmark
```

## Troubleshooting

### Common Issues

#### 1. Low Performance
**Symptoms:** TPS < 30
**Solutions:**
- Check CPU utilization: `htop`
- Verify SIMD support: `cat /proc/cpuinfo | grep avx`
- Increase thread count
- Enable quantization

#### 2. High Memory Usage
**Symptoms:** Out of memory errors
**Solutions:**
- Reduce cache sizes
- Enable more aggressive quantization
- Check for memory leaks
- Optimize memory layout

#### 3. Cache Misses
**Symptoms:** Low cache hit rate
**Solutions:**
- Increase cache sizes
- Optimize access patterns
- Enable prefetching
- Check memory alignment

#### 4. Quantization Errors
**Symptoms:** Incorrect output
**Solutions:**
- Disable quantization temporarily
- Check quantization parameters
- Verify model weights
- Use FP16 instead of INT8/INT4

### Performance Debugging

```bash
# Check system resources
lscpu
free -h
df -h

# Monitor performance
perf top -p $(pgrep inference_engine)

# Check memory usage
pmap $(pgrep inference_engine)

# Monitor cache performance
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./inference_engine benchmark
```

## Advanced Usage

### Custom Tokenization

```cpp
class CustomTokenizer {
public:
    std::vector<Token> encode(const std::string& text) {
        // Implement your tokenization logic
        return tokens;
    }
    
    std::string decode(const std::vector<Token>& tokens) {
        // Implement your detokenization logic
        return text;
    }
};
```

### Custom Quantization

```cpp
class CustomQuantizer : public QuantizationEngine {
public:
    void custom_quantize(const std::vector<Weight>& weights,
                        std::vector<QuantizedWeight>& quantized,
                        std::vector<float>& scales) {
        // Implement your quantization logic
    }
};
```

### Custom Caching

```cpp
class CustomCache : public CacheManager {
public:
    bool custom_get_cache(Token token, uint32_t position, CacheEntry& entry) {
        // Implement your caching logic
        return found;
    }
    
    void custom_set_cache(Token token, uint32_t position, const CacheEntry& entry) {
        // Implement your caching logic
    }
};
```

## Best Practices

### Development
1. Use performance counters
2. Profile regularly
3. Test on target hardware
4. Optimize hot paths
5. Monitor memory usage

### Deployment
1. Use release builds
2. Enable all optimizations
3. Configure for target hardware
4. Monitor performance metrics
5. Set up logging

### Maintenance
1. Regular performance testing
2. Update optimization strategies
3. Monitor hardware changes
4. Optimize for new workloads
5. Keep documentation updated

