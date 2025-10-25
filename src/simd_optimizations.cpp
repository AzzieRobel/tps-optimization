#include "inference_engine.h"
#include <immintrin.h>
#include <cstring>

// SIMD-optimized matrix multiplication using AVX2/AVX-512
void InferenceEngine::simd_matrix_multiply(const float* a, const float* b, float* c, 
                                          size_t m, size_t n, size_t k) {
    // Block size for cache optimization
    const size_t block_size = 64;
    
    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            for (size_t kk = 0; kk < k; kk += block_size) {
                // Process block
                size_t i_end = std::min(i + block_size, m);
                size_t j_end = std::min(j + block_size, n);
                size_t k_end = std::min(kk + block_size, k);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; jj += 8) { // Process 8 elements at once
                        __m256 sum = _mm256_setzero_ps();
                        
                        for (size_t kkk = kk; kkk < k_end; ++kkk) {
                            __m256 a_vec = _mm256_broadcast_ss(&a[ii * k + kkk]);
                            __m256 b_vec = _mm256_loadu_ps(&b[kkk * n + jj]);
                            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                        }
                        
                        _mm256_storeu_ps(&c[ii * n + jj], sum);
                    }
                }
            }
        }
    }
}

// SIMD-optimized attention score computation
void InferenceEngine::simd_attention_scores(const float* queries, const float* keys, 
                                           float* scores, size_t seq_len, size_t head_dim) {
    const size_t simd_width = 8; // AVX2 processes 8 floats at once
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; j += simd_width) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t d = 0; d < head_dim; d += simd_width) {
                __m256 q_vec = _mm256_loadu_ps(&queries[i * head_dim + d]);
                __m256 k_vec = _mm256_loadu_ps(&keys[j * head_dim + d]);
                sum = _mm256_fmadd_ps(q_vec, k_vec, sum);
            }
            
            // Horizontal sum
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), 
                                      _mm256_extractf128_ps(sum, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            
            float result = _mm_cvtss_f32(sum128);
            scores[i * seq_len + j] = result / sqrtf(head_dim);
        }
    }
}

// SIMD-optimized layer normalization
void InferenceEngine::simd_layer_norm(const float* input, float* output, size_t size, 
                                    float epsilon) {
    // Calculate mean using SIMD
    __m256 sum = _mm256_setzero_ps();
    size_t i;
    
    for (i = 0; i < size - 7; i += 8) {
        __m256 vec = _mm256_loadu_ps(&input[i]);
        sum = _mm256_add_ps(sum, vec);
    }
    
    // Handle remaining elements
    float mean = 0.0f;
    for (; i < size; ++i) {
        mean += input[i];
    }
    
    // Horizontal sum for SIMD part
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), 
                              _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    mean += _mm_cvtss_f32(sum128);
    mean /= size;
    
    // Calculate variance using SIMD
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 var_sum = _mm256_setzero_ps();
    
    for (i = 0; i < size - 7; i += 8) {
        __m256 vec = _mm256_loadu_ps(&input[i]);
        __m256 diff = _mm256_sub_ps(vec, mean_vec);
        var_sum = _mm256_fmadd_ps(diff, diff, var_sum);
    }
    
    // Handle remaining elements
    float variance = 0.0f;
    for (; i < size; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    
    // Horizontal sum for variance
    __m128 var128 = _mm_add_ps(_mm256_extractf128_ps(var_sum, 0), 
                              _mm256_extractf128_ps(var_sum, 1));
    var128 = _mm_hadd_ps(var128, var128);
    var128 = _mm_hadd_ps(var128, var128);
    variance += _mm_cvtss_f32(var128);
    variance /= size;
    
    float std_dev = sqrtf(variance + epsilon);
    __m256 std_vec = _mm256_set1_ps(std_dev);
    __m256 mean_vec_norm = _mm256_set1_ps(mean);
    
    // Apply normalization using SIMD
    for (i = 0; i < size - 7; i += 8) {
        __m256 vec = _mm256_loadu_ps(&input[i]);
        __m256 normalized = _mm256_div_ps(_mm256_sub_ps(vec, mean_vec_norm), std_vec);
        _mm256_storeu_ps(&output[i], normalized);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        output[i] = (input[i] - mean) / std_dev;
    }
}

// Quantized matrix multiplication using AVX2
void quantized_matrix_multiply_avx2(const int8_t* a, const int8_t* b, float* c,
                                   const float* a_scales, const float* b_scales,
                                   size_t m, size_t n, size_t k) {
    const size_t simd_width = 32; // AVX2 processes 32 int8 values at once
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m256i sum = _mm256_setzero_si256();
            
            for (size_t kk = 0; kk < k; kk += simd_width) {
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

// Memory prefetching for cache optimization
void prefetch_data(const void* ptr, size_t size) {
    const char* data = static_cast<const char*>(ptr);
    const size_t cache_line_size = 64;
    
    for (size_t i = 0; i < size; i += cache_line_size) {
        __builtin_prefetch(data + i, 0, 3); // Read, high temporal locality
    }
}

// Optimized memory copy with SIMD
void simd_memcpy(void* dest, const void* src, size_t n) {
    if (n < 32) {
        memcpy(dest, src, n);
        return;
    }
    
    char* d = static_cast<char*>(dest);
    const char* s = static_cast<const char*>(src);
    
    // Align destination to 32-byte boundary
    size_t align_offset = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
    if (align_offset > 0 && align_offset < n) {
        memcpy(d, s, align_offset);
        d += align_offset;
        s += align_offset;
        n -= align_offset;
    }
    
    // Copy 32-byte chunks using AVX2
    while (n >= 32) {
        __m256i data = _mm256_loadu_si256((__m256i*)s);
        _mm256_storeu_si256((__m256i*)d, data);
        d += 32;
        s += 32;
        n -= 32;
    }
    
    // Handle remaining bytes
    if (n > 0) {
        memcpy(d, s, n);
    }
}

// Optimized activation functions using SIMD
void simd_gelu(const float* input, float* output, size_t size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    __m256 sqrt_2_over_pi_vec = _mm256_set1_ps(sqrt_2_over_pi);
    __m256 coeff_vec = _mm256_set1_ps(coeff);
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 half_vec = _mm256_set1_ps(0.5f);
    
    for (size_t i = 0; i < size - 7; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 x_cubed = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
        __m256 inner = _mm256_fmadd_ps(coeff_vec, x_cubed, x);
        __m256 tanh_inner = _mm256_mul_ps(sqrt_2_over_pi_vec, inner);
        
        // Approximate tanh using polynomial
        __m256 tanh_result = _mm256_mul_ps(tanh_inner, 
            _mm256_fmadd_ps(tanh_inner, 
                _mm256_fmadd_ps(tanh_inner, 
                    _mm256_fmadd_ps(tanh_inner, 
                        _mm256_set1_ps(0.16666667f), 
                        _mm256_set1_ps(0.33333333f)), 
                    _mm256_set1_ps(0.5f)), 
                one_vec));
        
        __m256 result = _mm256_mul_ps(half_vec, 
            _mm256_mul_ps(x, _mm256_add_ps(one_vec, tanh_result)));
        
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = (size / 8) * 8; i < size; ++i) {
        float x = input[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Optimized softmax using SIMD
void simd_softmax(const float* input, float* output, size_t size) {
    // Find maximum value for numerical stability
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    
    for (size_t i = 0; i < size - 7; i += 8) {
        __m256 vec = _mm256_loadu_ps(&input[i]);
        max_vec = _mm256_max_ps(max_vec, vec);
    }
    
    // Handle remaining elements
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = (size / 8) * 8; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Get maximum from SIMD
    __m128 max128 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 0),
                               _mm256_extractf128_ps(max_vec, 1));
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_movehl_ps(max128, max128));
    float simd_max = _mm_cvtss_f32(max128);
    
    max_val = std::max(max_val, simd_max);
    __m256 max_val_vec = _mm256_set1_ps(max_val);
    
    // Calculate exponentials and sum
    __m256 sum_vec = _mm256_setzero_ps();
    std::vector<__m256> exp_vecs;
    
    for (size_t i = 0; i < size - 7; i += 8) {
        __m256 vec = _mm256_loadu_ps(&input[i]);
        __m256 exp_vec = _mm256_exp_ps(_mm256_sub_ps(vec, max_val_vec));
        exp_vecs.push_back(exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }
    
    // Handle remaining elements
    float sum = 0.0f;
    for (size_t i = (size / 8) * 8; i < size; ++i) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Get sum from SIMD
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                              _mm256_extractf128_ps(sum_vec, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float simd_sum = _mm_cvtss_f32(sum128);
    
    sum += simd_sum;
    __m256 sum_vec_final = _mm256_set1_ps(sum);
    
    // Normalize
    size_t vec_idx = 0;
    for (size_t i = 0; i < size - 7; i += 8) {
        __m256 normalized = _mm256_div_ps(exp_vecs[vec_idx], sum_vec_final);
        _mm256_storeu_ps(&output[i], normalized);
        vec_idx++;
    }
    
    // Handle remaining elements
    for (size_t i = (size / 8) * 8; i < size; ++i) {
        output[i] /= sum;
    }
}

