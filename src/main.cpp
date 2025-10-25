#include "inference_engine.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

// Tokenizer for text processing
class SimpleTokenizer {
public:
    SimpleTokenizer() {
        // Initialize with basic vocabulary
        vocab_.reserve(50257);
        for (int i = 0; i < 50257; ++i) {
            vocab_[i] = "token_" + std::to_string(i);
        }
    }
    
    std::vector<Token> encode(const std::string& text) {
        // Simple word-based tokenization (in reality, would use proper BPE)
        std::vector<Token> tokens;
        std::string word;
        
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!word.empty()) {
                    tokens.push_back(hash_word(word) % 50257);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        
        if (!word.empty()) {
            tokens.push_back(hash_word(word) % 50257);
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<Token>& tokens) {
        std::string result;
        for (Token token : tokens) {
            if (token < vocab_.size()) {
                result += vocab_[token] + " ";
            }
        }
        return result;
    }
    
private:
    std::vector<std::string> vocab_;
    
    uint32_t hash_word(const std::string& word) {
        uint32_t hash = 0;
        for (char c : word) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};

// Performance benchmarking
class Benchmark {
public:
    static void run_performance_test(InferenceEngine& engine, const std::string& prompt) {
        std::cout << "Running performance benchmark..." << std::endl;
        std::cout << "Target: >30 TPS" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        SimpleTokenizer tokenizer;
        auto prompt_tokens = tokenizer.encode(prompt);
        
        // Warmup
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            engine.generate(prompt_tokens, 10, 0.7f);
        }
        
        // Performance test
        const int num_runs = 10;
        const int tokens_per_run = 100;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int run = 0; run < num_runs; ++run) {
            auto result = engine.generate(prompt_tokens, tokens_per_run, 0.7f);
            std::cout << "Run " << (run + 1) << "/" << num_runs 
                      << " - Generated " << result.size() << " tokens" << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Calculate performance metrics
        double total_tokens = num_runs * tokens_per_run;
        double total_time_seconds = total_duration.count() / 1000.0;
        double tps = total_tokens / total_time_seconds;
        
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "Performance Results:" << std::endl;
        std::cout << "Total tokens generated: " << total_tokens << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) 
                  << total_time_seconds << " seconds" << std::endl;
        std::cout << "Tokens per second: " << std::fixed << std::setprecision(2) << tps << std::endl;
        std::cout << "Target achieved: " << (tps > 30.0 ? "YES" : "NO") << std::endl;
        
        // Memory usage
        auto stats = engine.get_stats();
        std::cout << "Cache hits: " << stats.cache_hits.load() << std::endl;
        std::cout << "Cache misses: " << stats.cache_misses.load() << std::endl;
        std::cout << "Quantized operations: " << stats.quantized_ops.load() << std::endl;
    }
    
    static void run_memory_test(InferenceEngine& engine) {
        std::cout << "\nRunning memory optimization test..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        SimpleTokenizer tokenizer;
        std::string test_prompt = "The quick brown fox jumps over the lazy dog";
        auto prompt_tokens = tokenizer.encode(test_prompt);
        
        // Test memory efficiency with long sequences
        std::vector<int> sequence_lengths = {100, 500, 1000, 2000};
        
        for (int length : sequence_lengths) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = engine.generate(prompt_tokens, length, 0.7f);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            double tps = result.size() / (duration.count() / 1000.0);
            
            std::cout << "Sequence length " << length << ": " 
                      << std::fixed << std::setprecision(2) << tps << " TPS" << std::endl;
        }
    }
};

// Interactive mode
void interactive_mode(InferenceEngine& engine) {
    std::cout << "\nInteractive mode - Type 'quit' to exit" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    SimpleTokenizer tokenizer;
    std::string input;
    
    while (true) {
        std::cout << "\nEnter prompt: ";
        std::getline(std::cin, input);
        
        if (input == "quit") break;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto prompt_tokens = tokenizer.encode(input);
        auto result = engine.generate(prompt_tokens, 50, 0.7f);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double tps = result.size() / (duration.count() / 1000.0);
        
        std::cout << "Generated: " << tokenizer.decode(result) << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << tps << " TPS" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "High-Performance GPT-OSS-20B Inference Engine" << std::endl;
    std::cout << "Target: >30 TPS Performance" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Initialize inference engine
    InferenceEngine engine;
    
    // Load model (placeholder - in reality would load actual model)
    std::cout << "Loading model..." << std::endl;
    if (!engine.load_model("gpt-oss-20b.bin")) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    // Initialize quantization
    std::cout << "Initializing quantization..." << std::endl;
    if (!engine.initialize_quantization()) {
        std::cerr << "Failed to initialize quantization!" << std::endl;
        return 1;
    }
    
    // Configure engine
    engine.set_num_threads(std::thread::hardware_concurrency());
    engine.set_cache_size(1024 * 1024 * 1024); // 1GB cache
    engine.enable_quantization(true);
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    
    // Parse command line arguments
    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "benchmark") {
            std::string prompt = "The future of artificial intelligence";
            if (argc > 2) {
                prompt = argv[2];
            }
            Benchmark::run_performance_test(engine, prompt);
            Benchmark::run_memory_test(engine);
        } else if (mode == "interactive") {
            interactive_mode(engine);
        } else if (mode == "generate") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " generate <prompt>" << std::endl;
                return 1;
            }
            
            std::string prompt = argv[2];
            SimpleTokenizer tokenizer;
            auto prompt_tokens = tokenizer.encode(prompt);
            auto result = engine.generate(prompt_tokens, 100, 0.7f);
            
            std::cout << "Generated text: " << tokenizer.decode(result) << std::endl;
        }
    } else {
        // Default: run benchmark
        Benchmark::run_performance_test(engine, "The future of artificial intelligence");
    }
    
    // Print final statistics
    auto stats = engine.get_stats();
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Total tokens generated: " << stats.tokens_generated.load() << std::endl;
    std::cout << "Average TPS: " << std::fixed << std::setprecision(2) << stats.get_tps() << std::endl;
    std::cout << "Cache hit rate: " << std::fixed << std::setprecision(2) 
              << (100.0 * stats.cache_hits.load() / (stats.cache_hits.load() + stats.cache_misses.load())) 
              << "%" << std::endl;
    
    return 0;
}

