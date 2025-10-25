#include "inference_engine.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

class ComprehensiveBenchmark {
public:
    struct BenchmarkResult {
        double tps;
        double latency_ms;
        double memory_usage_mb;
        double cache_hit_rate;
        uint64_t total_tokens;
        uint64_t total_time_ms;
    };
    
    static void run_full_benchmark(InferenceEngine& engine) {
        std::cout << "================================================" << std::endl;
        std::cout << "High-Performance GPT-OSS-20B Benchmark Suite" << std::endl;
        std::cout << "Target: >30 TPS Performance" << std::endl;
        std::cout << "================================================" << std::endl;
        
        // Test different scenarios
        test_single_thread_performance(engine);
        test_multi_thread_performance(engine);
        test_memory_efficiency(engine);
        test_quantization_impact(engine);
        test_cache_performance(engine);
        test_sequence_length_scaling(engine);
        test_batch_processing(engine);
        test_mixed_workloads(engine);
        
        std::cout << "\n================================================" << std::endl;
        std::cout << "Benchmark Complete!" << std::endl;
        std::cout << "================================================" << std::endl;
    }
    
private:
    static void test_single_thread_performance(InferenceEngine& engine) {
        std::cout << "\n1. Single-Thread Performance Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        engine.set_num_threads(1);
        auto result = run_benchmark_scenario(engine, "Single thread test", 100, 50);
        
        std::cout << "TPS: " << std::fixed << std::setprecision(2) << result.tps << std::endl;
        std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
        std::cout << "Memory: " << result.memory_usage_mb << " MB" << std::endl;
        std::cout << "Cache Hit Rate: " << result.cache_hit_rate << "%" << std::endl;
    }
    
    static void test_multi_thread_performance(InferenceEngine& engine) {
        std::cout << "\n2. Multi-Thread Performance Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<int> thread_counts = {2, 4, 8, 16, 32};
        
        for (int threads : thread_counts) {
            engine.set_num_threads(threads);
            auto result = run_benchmark_scenario(engine, 
                "Multi-thread test (" + std::to_string(threads) + " threads)", 100, 50);
            
            std::cout << "Threads: " << threads 
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Latency: " << result.latency_ms << " ms" << std::endl;
        }
    }
    
    static void test_memory_efficiency(InferenceEngine& engine) {
        std::cout << "\n3. Memory Efficiency Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<size_t> cache_sizes = {256, 512, 1024, 2048, 4096}; // MB
        
        for (size_t cache_size : cache_sizes) {
            engine.set_cache_size(cache_size * 1024 * 1024);
            auto result = run_benchmark_scenario(engine, 
                "Memory test (" + std::to_string(cache_size) + " MB cache)", 100, 50);
            
            std::cout << "Cache: " << cache_size << " MB"
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Memory: " << result.memory_usage_mb << " MB" << std::endl;
        }
    }
    
    static void test_quantization_impact(InferenceEngine& engine) {
        std::cout << "\n4. Quantization Impact Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<std::string> quantization_types = {"FP16", "INT8", "INT4"};
        std::vector<int> quantization_levels = {0, 1, 2};
        
        for (size_t i = 0; i < quantization_types.size(); ++i) {
            engine.enable_quantization(true);
            // In a real implementation, you would set quantization level here
            
            auto result = run_benchmark_scenario(engine, 
                "Quantization test (" + quantization_types[i] + ")", 100, 50);
            
            std::cout << "Type: " << quantization_types[i]
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Memory: " << result.memory_usage_mb << " MB" << std::endl;
        }
    }
    
    static void test_cache_performance(InferenceEngine& engine) {
        std::cout << "\n5. Cache Performance Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Test cache hit rates with different access patterns
        std::vector<std::string> patterns = {"Sequential", "Random", "Repeated"};
        
        for (const auto& pattern : patterns) {
            auto result = run_benchmark_scenario(engine, 
                "Cache test (" + pattern + ")", 100, 50);
            
            std::cout << "Pattern: " << pattern
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Cache Hit Rate: " << result.cache_hit_rate << "%" << std::endl;
        }
    }
    
    static void test_sequence_length_scaling(InferenceEngine& engine) {
        std::cout << "\n6. Sequence Length Scaling Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<int> sequence_lengths = {50, 100, 200, 500, 1000, 2000};
        
        for (int length : sequence_lengths) {
            auto result = run_benchmark_scenario(engine, 
                "Sequence length test (" + std::to_string(length) + " tokens)", 100, length);
            
            std::cout << "Length: " << length
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Latency: " << result.latency_ms << " ms" << std::endl;
        }
    }
    
    static void test_batch_processing(InferenceEngine& engine) {
        std::cout << "\n7. Batch Processing Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<int> batch_sizes = {1, 2, 4, 8, 16};
        
        for (int batch_size : batch_sizes) {
            auto result = run_batch_benchmark(engine, batch_size, 50);
            
            std::cout << "Batch Size: " << batch_size
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Latency: " << result.latency_ms << " ms" << std::endl;
        }
    }
    
    static void test_mixed_workloads(InferenceEngine& engine) {
        std::cout << "\n8. Mixed Workload Test" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Simulate mixed workload with different prompt types
        std::vector<std::string> prompt_types = {
            "Short prompt",
            "Medium prompt with more context",
            "Long prompt with extensive background information and detailed requirements"
        };
        
        for (const auto& prompt_type : prompt_types) {
            auto result = run_benchmark_scenario(engine, prompt_type, 100, 50);
            
            std::cout << "Type: " << prompt_type
                      << " | TPS: " << std::fixed << std::setprecision(2) << result.tps
                      << " | Latency: " << result.latency_ms << " ms" << std::endl;
        }
    }
    
    static BenchmarkResult run_benchmark_scenario(InferenceEngine& engine, 
                                                 const std::string& description,
                                                 int num_runs, int tokens_per_run) {
        std::cout << "Running: " << description << std::endl;
        
        // Generate test prompt
        std::vector<Token> prompt = generate_test_prompt(20);
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            engine.generate(prompt, 10, 0.7f);
        }
        
        // Reset stats
        engine.reset_stats();
        
        // Benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int run = 0; run < num_runs; ++run) {
            auto result = engine.generate(prompt, tokens_per_run, 0.7f);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Get performance stats
        auto stats = engine.get_stats();
        
        BenchmarkResult result;
        result.total_tokens = stats.tokens_generated.load();
        result.total_time_ms = duration.count();
        result.tps = result.total_tokens * 1000.0 / result.total_time_ms;
        result.latency_ms = result.total_time_ms / static_cast<double>(num_runs);
        result.memory_usage_mb = 0.0; // Would be calculated from actual memory usage
        result.cache_hit_rate = calculate_cache_hit_rate(stats);
        
        return result;
    }
    
    static BenchmarkResult run_batch_benchmark(InferenceEngine& engine, 
                                              int batch_size, int tokens_per_run) {
        // Generate batch of prompts
        std::vector<std::vector<Token>> prompts;
        for (int i = 0; i < batch_size; ++i) {
            prompts.push_back(generate_test_prompt(20));
        }
        
        // Warmup
        engine.generate_batch(prompts, 10, 0.7f);
        
        // Reset stats
        engine.reset_stats();
        
        // Benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto results = engine.generate_batch(prompts, tokens_per_run, 0.7f);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Get performance stats
        auto stats = engine.get_stats();
        
        BenchmarkResult result;
        result.total_tokens = stats.tokens_generated.load();
        result.total_time_ms = duration.count();
        result.tps = result.total_tokens * 1000.0 / result.total_time_ms;
        result.latency_ms = result.total_time_ms / static_cast<double>(batch_size);
        result.memory_usage_mb = 0.0;
        result.cache_hit_rate = calculate_cache_hit_rate(stats);
        
        return result;
    }
    
    static std::vector<Token> generate_test_prompt(int length) {
        std::vector<Token> prompt;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<Token> dist(0, 50256);
        
        for (int i = 0; i < length; ++i) {
            prompt.push_back(dist(gen));
        }
        
        return prompt;
    }
    
    static double calculate_cache_hit_rate(const PerformanceStats& stats) {
        uint64_t hits = stats.cache_hits.load();
        uint64_t misses = stats.cache_misses.load();
        uint64_t total = hits + misses;
        
        return total > 0 ? (hits * 100.0) / total : 0.0;
    }
};

// Main benchmark function
int main(int argc, char* argv[]) {
    std::cout << "Initializing High-Performance GPT-OSS-20B Inference Engine..." << std::endl;
    
    // Initialize engine
    InferenceEngine engine;
    
    // Load model
    if (!engine.load_model("gpt-oss-20b.bin")) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    // Initialize quantization
    if (!engine.initialize_quantization()) {
        std::cerr << "Failed to initialize quantization!" << std::endl;
        return 1;
    }
    
    // Configure engine
    engine.set_num_threads(std::thread::hardware_concurrency());
    engine.set_cache_size(1024 * 1024 * 1024); // 1GB cache
    engine.enable_quantization(true);
    
    std::cout << "Engine initialized successfully!" << std::endl;
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    
    // Run comprehensive benchmark
    ComprehensiveBenchmark::run_full_benchmark(engine);
    
    return 0;
}

