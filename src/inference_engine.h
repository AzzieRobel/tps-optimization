#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

// Core data types optimized for performance
using Token = uint32_t;
using Weight = float;
using QuantizedWeight = int8_t;

// Performance counters
struct PerformanceStats {
    std::atomic<uint64_t> tokens_generated{0};
    std::atomic<uint64_t> total_time_ms{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> quantized_ops{0};
    std::atomic<uint64_t> memory_allocations{0};
    
    double get_tps() const {
        uint64_t tokens = tokens_generated.load();
        uint64_t time = total_time_ms.load();
        return time > 0 ? (tokens * 1000.0) / time : 0.0;
    }
};

// Memory-mapped model weights
struct ModelWeights {
    std::vector<Weight> attention_weights;
    std::vector<Weight> feedforward_weights;
    std::vector<Weight> embedding_weights;
    std::vector<Weight> output_weights;
    
    // Quantized versions for performance
    std::vector<QuantizedWeight> quantized_attention;
    std::vector<QuantizedWeight> quantized_feedforward;
    std::vector<QuantizedWeight> quantized_embedding;
    std::vector<QuantizedWeight> quantized_output;
    
    // Quantization parameters
    std::vector<float> attention_scales;
    std::vector<float> feedforward_scales;
    std::vector<float> embedding_scales;
    std::vector<float> output_scales;
};

// Multi-level cache system
class CacheManager {
public:
    struct CacheEntry {
        std::vector<float> key_cache;
        std::vector<float> value_cache;
        std::vector<float> attention_scores;
        uint64_t timestamp;
        uint32_t sequence_length;
    };
    
    CacheManager(size_t l1_size = 1024 * 1024, size_t l2_size = 1024 * 1024 * 1024);
    ~CacheManager();
    
    bool get_cache(Token token, uint32_t position, CacheEntry& entry);
    void set_cache(Token token, uint32_t position, const CacheEntry& entry);
    void evict_old_entries();
    
private:
    std::unordered_map<uint64_t, CacheEntry> l1_cache_;
    std::unordered_map<uint64_t, CacheEntry> l2_cache_;
    size_t l1_max_size_;
    size_t l2_max_size_;
    std::mutex cache_mutex_;
};

// Quantization engine for dynamic precision optimization
class QuantizationEngine {
public:
    QuantizationEngine();
    
    void quantize_weights(const std::vector<Weight>& weights, 
                         std::vector<QuantizedWeight>& quantized,
                         std::vector<float>& scales);
    
    void dequantize_weights(const std::vector<QuantizedWeight>& quantized,
                           const std::vector<float>& scales,
                           std::vector<Weight>& weights);
    
    // Dynamic quantization based on activation patterns
    void adaptive_quantize(const std::vector<Weight>& weights,
                          const std::vector<float>& activations,
                          std::vector<QuantizedWeight>& quantized,
                          std::vector<float>& scales);
    
private:
    float find_optimal_scale(const std::vector<Weight>& weights);
    void apply_symmetric_quantization(const std::vector<Weight>& weights,
                                     std::vector<QuantizedWeight>& quantized,
                                     float scale);
};

// Memory manager with paging support
class MemoryManager {
public:
    MemoryManager(size_t total_memory = 32ULL * 1024 * 1024 * 1024);
    ~MemoryManager();
    
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);
    void* reallocate(void* ptr, size_t new_size);
    
    // Memory mapping for model weights
    void* map_model_file(const std::string& filename, size_t offset, size_t length);
    void unmap_memory(void* ptr, size_t length);
    
    // Paging support
    void enable_paging(bool enable);
    void set_page_size(size_t page_size);
    
private:
    size_t total_memory_;
    size_t used_memory_;
    size_t page_size_;
    bool paging_enabled_;
    std::mutex memory_mutex_;
    std::unordered_map<void*, size_t> allocations_;
};

// Core inference engine
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Model loading and initialization
    bool load_model(const std::string& model_path);
    bool initialize_quantization();
    
    // Inference methods
    std::vector<Token> generate(const std::vector<Token>& prompt, 
                               uint32_t max_tokens = 512,
                               float temperature = 0.7f);
    
    // Batch processing for higher throughput
    std::vector<std::vector<Token>> generate_batch(
        const std::vector<std::vector<Token>>& prompts,
        uint32_t max_tokens = 512,
        float temperature = 0.7f);
    
    // Performance monitoring
    PerformanceStats get_stats() const;
    void reset_stats();
    
    // Configuration
    void set_num_threads(uint32_t threads);
    void set_cache_size(size_t size);
    void enable_quantization(bool enable);
    
private:
    // Core components
    std::unique_ptr<ModelWeights> model_weights_;
    std::unique_ptr<CacheManager> cache_manager_;
    std::unique_ptr<QuantizationEngine> quantizer_;
    std::unique_ptr<MemoryManager> memory_manager_;
    
    // Performance tracking
    PerformanceStats stats_;
    
    // Threading
    uint32_t num_threads_;
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::atomic<bool> should_stop_;
    
    // Internal methods
    void worker_thread_function();
    std::vector<float> forward_pass(const std::vector<Token>& tokens, uint32_t position);
    std::vector<float> attention_layer(const std::vector<float>& input, uint32_t layer);
    std::vector<float> feedforward_layer(const std::vector<float>& input, uint32_t layer);
    Token sample_token(const std::vector<float>& logits, float temperature);
    
    // SIMD-optimized operations
    void simd_matrix_multiply(const float* a, const float* b, float* c, 
                              size_t m, size_t n, size_t k);
    void simd_attention_scores(const float* queries, const float* keys, 
                               float* scores, size_t seq_len, size_t head_dim);
    void simd_layer_norm(const float* input, float* output, size_t size, 
                        float epsilon = 1e-5f);
    
    // Memory optimization
    void prefetch_weights(uint32_t layer);
    void evict_unused_weights();
    void optimize_memory_layout();
};

