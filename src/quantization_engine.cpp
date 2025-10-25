#include "inference_engine.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

// Advanced quantization strategies for maximum performance

class AdvancedQuantizationEngine {
public:
    // Dynamic quantization based on activation patterns
    void dynamic_quantize(const std::vector<Weight>& weights,
                         const std::vector<float>& activations,
                         std::vector<QuantizedWeight>& quantized,
                         std::vector<float>& scales,
                         QuantizationType type = INT8) {
        
        switch (type) {
            case INT8:
                quantize_int8(weights, activations, quantized, scales);
                break;
            case INT4:
                quantize_int4(weights, activations, quantized, scales);
                break;
            case FP16:
                quantize_fp16(weights, quantized, scales);
                break;
        }
    }
    
    // INT8 quantization with calibration
    void quantize_int8(const std::vector<Weight>& weights,
                      const std::vector<float>& activations,
                      std::vector<QuantizedWeight>& quantized,
                      std::vector<float>& scales) {
        
        const size_t block_size = 256;
        quantized.resize(weights.size());
        scales.resize(weights.size() / block_size);
        
        for (size_t i = 0; i < weights.size(); i += block_size) {
            size_t current_block = std::min(block_size, weights.size() - i);
            
            // Calculate optimal scale for this block
            float weight_scale = calculate_optimal_scale(
                weights.data() + i, current_block);
            
            float activation_scale = 1.0f;
            if (i < activations.size()) {
                activation_scale = std::abs(activations[i]);
            }
            
            float combined_scale = weight_scale * activation_scale;
            scales[i / block_size] = combined_scale;
            
            // Quantize using SIMD
            quantize_block_simd(weights.data() + i, 
                               quantized.data() + i,
                               current_block, combined_scale);
        }
    }
    
    // INT4 quantization for extreme compression
    void quantize_int4(const std::vector<Weight>& weights,
                      const std::vector<float>& activations,
                      std::vector<QuantizedWeight>& quantized,
                      std::vector<float>& scales) {
        
        const size_t block_size = 128; // Smaller blocks for INT4
        quantized.resize(weights.size() / 2); // Packed storage
        scales.resize(weights.size() / block_size);
        
        for (size_t i = 0; i < weights.size(); i += block_size) {
            size_t current_block = std::min(block_size, weights.size() - i);
            
            float scale = calculate_optimal_scale(weights.data() + i, current_block);
            scales[i / block_size] = scale;
            
            // Pack two INT4 values into one INT8
            for (size_t j = 0; j < current_block; j += 2) {
                int8_t val1 = static_cast<int8_t>(std::max(-8, std::min(7, 
                    static_cast<int>(weights[i + j] / scale))));
                int8_t val2 = static_cast<int8_t>(std::max(-8, std::min(7, 
                    static_cast<int>(weights[i + j + 1] / scale))));
                
                quantized[i / 2 + j / 2] = (val1 & 0xF) | ((val2 & 0xF) << 4);
            }
        }
    }
    
    // FP16 quantization for balanced performance/accuracy
    void quantize_fp16(const std::vector<Weight>& weights,
                      std::vector<QuantizedWeight>& quantized,
                      std::vector<float>& scales) {
        
        quantized.resize(weights.size() * 2); // FP16 is 2 bytes
        scales.resize(1); // Single scale for FP16
        scales[0] = 1.0f;
        
        // Convert to FP16 using SIMD
        for (size_t i = 0; i < weights.size(); i += 8) {
            __m256 fp32_vec = _mm256_loadu_ps(&weights[i]);
            __m128i fp16_vec = _mm256_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i*)&quantized[i * 2], fp16_vec);
        }
    }
    
private:
    enum QuantizationType { INT8, INT4, FP16 };
    
    float calculate_optimal_scale(const float* weights, size_t count) {
        if (count == 0) return 1.0f;
        
        // Use SIMD to find min/max efficiently
        __m256 min_vec = _mm256_set1_ps(std::numeric_limits<float>::max());
        __m256 max_vec = _mm256_set1_ps(std::numeric_limits<float>::lowest());
        
        size_t i;
        for (i = 0; i < count - 7; i += 8) {
            __m256 vec = _mm256_loadu_ps(&weights[i]);
            min_vec = _mm256_min_ps(min_vec, vec);
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        
        // Handle remaining elements
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (; i < count; ++i) {
            min_val = std::min(min_val, weights[i]);
            max_val = std::max(max_val, weights[i]);
        }
        
        // Get min/max from SIMD
        __m128 min128 = _mm_min_ps(_mm256_extractf128_ps(min_vec, 0),
                                  _mm256_extractf128_ps(min_vec, 1));
        min128 = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
        min128 = _mm_min_ss(min128, _mm_movehl_ps(min128, min128));
        float simd_min = _mm_cvtss_f32(min128);
        
        __m128 max128 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 0),
                                  _mm256_extractf128_ps(max_vec, 1));
        max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        max128 = _mm_max_ss(max128, _mm_movehl_ps(max128, max128));
        float simd_max = _mm_cvtss_f32(max128);
        
        min_val = std::min(min_val, simd_min);
        max_val = std::max(max_val, simd_max);
        
        float range = std::max(std::abs(min_val), std::abs(max_val));
        return range / 127.0f; // INT8 range
    }
    
    void quantize_block_simd(const float* weights, int8_t* quantized,
                            size_t count, float scale) {
        __m256 scale_vec = _mm256_set1_ps(scale);
        __m256 min_vec = _mm256_set1_ps(-128.0f);
        __m256 max_vec = _mm256_set1_ps(127.0f);
        
        for (size_t i = 0; i < count - 7; i += 8) {
            __m256 weight_vec = _mm256_loadu_ps(&weights[i]);
            __m256 scaled_vec = _mm256_div_ps(weight_vec, scale_vec);
            __m256 clamped_vec = _mm256_max_ps(min_vec, 
                _mm256_min_ps(max_vec, scaled_vec));
            
            __m256i int_vec = _mm256_cvtps_epi32(clamped_vec);
            __m128i int8_vec = _mm256_cvtepi32_epi8(int_vec);
            _mm_storel_epi64((__m128i*)&quantized[i], int8_vec);
        }
        
        // Handle remaining elements
        for (size_t i = (count / 8) * 8; i < count; ++i) {
            quantized[i] = static_cast<int8_t>(std::max(-128, 
                std::min(127, static_cast<int>(weights[i] / scale))));
        }
    }
};

// Quantized matrix multiplication kernels
class QuantizedKernels {
public:
    // INT8 matrix multiplication using AVX2
    static void int8_matmul_avx2(const int8_t* a, const int8_t* b, float* c,
                                const float* a_scales, const float* b_scales,
                                size_t m, size_t n, size_t k) {
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                __m256i sum = _mm256_setzero_si256();
                
                for (size_t kk = 0; kk < k; kk += 32) {
                    __m256i a_vec = _mm256_loadu_si256((__m256i*)&a[i * k + kk]);
                    __m256i b_vec = _mm256_loadu_si256((__m256i*)&b[kk * n + j]);
                    
                    // Multiply and accumulate
                    __m256i prod = _mm256_maddubs_epi16(a_vec, b_vec);
                    sum = _mm256_add_epi16(sum, prod);
                }
                
                // Horizontal sum
                __m128i sum128 = _mm_add_epi16(_mm256_extracti128_si256(sum, 0),
                                              _mm256_extracti128_si256(sum, 1));
                sum128 = _mm_sad_epu8(sum128, _mm_setzero_si128());
                sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                
                int32_t result = _mm_cvtsi128_si32(sum128);
                c[i * n + j] = result * a_scales[i] * b_scales[j];
            }
        }
    }
    
    // INT4 matrix multiplication using AVX2
    static void int4_matmul_avx2(const int8_t* a, const int8_t* b, float* c,
                                const float* a_scales, const float* b_scales,
                                size_t m, size_t n, size_t k) {
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                int32_t sum = 0;
                
                for (size_t kk = 0; kk < k; kk += 2) {
                    // Unpack INT4 values
                    int8_t a_val1 = a[i * k + kk] & 0xF;
                    int8_t a_val2 = (a[i * k + kk] >> 4) & 0xF;
                    int8_t b_val1 = b[kk * n + j] & 0xF;
                    int8_t b_val2 = (b[kk * n + j] >> 4) & 0xF;
                    
                    sum += a_val1 * b_val1 + a_val2 * b_val2;
                }
                
                c[i * n + j] = sum * a_scales[i] * b_scales[j];
            }
        }
    }
    
    // FP16 matrix multiplication using AVX2
    static void fp16_matmul_avx2(const int8_t* a, const int8_t* b, float* c,
                                const float* a_scales, const float* b_scales,
                                size_t m, size_t n, size_t k) {
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                
                for (size_t kk = 0; kk < k; ++kk) {
                    // Load FP16 values and convert to FP32
                    __m128i a_fp16 = _mm_loadu_si128((__m128i*)&a[i * k + kk]);
                    __m128i b_fp16 = _mm_loadu_si128((__m128i*)&b[kk * n + j]);
                    
                    __m256 a_fp32 = _mm256_cvtph_ps(a_fp16);
                    __m256 b_fp32 = _mm256_cvtph_ps(b_fp16);
                    
                    sum = _mm256_fmadd_ps(a_fp32, b_fp32, sum);
                }
                
                _mm256_storeu_ps(&c[i * n + j], sum);
            }
        }
    }
};

// Adaptive quantization based on runtime performance
class AdaptiveQuantization {
public:
    AdaptiveQuantization() : performance_threshold_(30.0f) {}
    
    void update_quantization_strategy(const std::vector<float>& activations,
                                    float current_tps) {
        if (current_tps < performance_threshold_) {
            // Increase quantization aggressiveness
            quantization_level_ = std::min(3, quantization_level_ + 1);
        } else {
            // Maintain or reduce quantization for accuracy
            quantization_level_ = std::max(0, quantization_level_ - 1);
        }
        
        // Update quantization parameters based on activation patterns
        update_quantization_parameters(activations);
    }
    
    QuantizationType get_optimal_quantization() const {
        switch (quantization_level_) {
            case 0: return FP16;
            case 1: return INT8;
            case 2: return INT4;
            case 3: return INT4; // Most aggressive
            default: return INT8;
        }
    }
    
private:
    float performance_threshold_;
    int quantization_level_ = 1; // Start with INT8
    
    void update_quantization_parameters(const std::vector<float>& activations) {
        // Analyze activation patterns to optimize quantization
        float mean_activation = 0.0f;
        float max_activation = 0.0f;
        
        for (float activation : activations) {
            mean_activation += activation;
            max_activation = std::max(max_activation, std::abs(activation));
        }
        
        mean_activation /= activations.size();
        
        // Adjust quantization based on activation characteristics
        if (max_activation > 10.0f) {
            // High dynamic range - use more conservative quantization
            quantization_level_ = std::max(0, quantization_level_ - 1);
        } else if (max_activation < 1.0f) {
            // Low dynamic range - can use more aggressive quantization
            quantization_level_ = std::min(3, quantization_level_ + 1);
        }
    }
};

