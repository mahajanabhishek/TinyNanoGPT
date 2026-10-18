#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cassert>

// Simple tensor class for float data
class Tensor {
public:
    std::vector<int64_t> dims;
    std::vector<float> data;
    
    Tensor() = default;
    
    Tensor(std::vector<int64_t> dims) : dims(dims) {
        int64_t size = 1;
        for (int64_t d : dims) size *= d;
        data.resize(size);
    }
    
    Tensor(std::vector<int64_t> dims, std::vector<float> data) : dims(dims), data(data) {
        int64_t size = 1;
        for (int64_t d : dims) size *= d;
        assert(data.size() == size);
    }
    
    // Initialize with random values from normal distribution
    void randn(float mean = 0.0f, float std = 0.02f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(mean, std);
        for (float& v : data) v = dist(gen);
    }
    
    // Initialize with uniform distribution
    void uniform(float a = -0.1f, float b = 0.1f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(a, b);
        for (float& v : data) v = dist(gen);
    }
    
    // Fill with constant
    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }
    
    // Get element at index (for 1D tensor)
    float& operator[](int64_t idx) {
        return data[idx];
    }
    
    const float& operator[](int64_t idx) const {
        return data[idx];
    }
    
    // Get element at multi-dimensional index
    float& at(const std::vector<int64_t>& indices) {
        int64_t idx = 0;
        int64_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
    
    const float& at(const std::vector<int64_t>& indices) const {
        int64_t idx = 0;
        int64_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
    
    // Reshape tensor
    void reshape(std::vector<int64_t> new_dims) {
        int64_t new_size = 1;
        for (int64_t d : new_dims) new_size *= d;
        int64_t old_size = 1;
        for (int64_t d : dims) old_size *= d;
        assert(new_size == old_size);
        dims = new_dims;
    }
    
    // Get total number of elements
    int64_t size() const {
        int64_t s = 1;
        for (int64_t d : dims) s *= d;
        return s;
    }
    
    // Print tensor shape and first few elements
    void print_shape(const std::string& name = "") const {
        std::cout << name << " shape: [";
        for (size_t i = 0; i < dims.size(); i++) {
            std::cout << dims[i];
            if (i < dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

// Basic operations
namespace ops {

// Element-wise addition: c = a + b
Tensor add(const Tensor& a, const Tensor& b) {
    std::cerr << "add: a.dims = [";
    for (size_t i = 0; i < a.dims.size(); ++i) {
        std::cerr << a.dims[i];
        if (i + 1 < a.dims.size()) std::cerr << ", ";
    }
    std::cerr << "], b.dims = [";
    for (size_t i = 0; i < b.dims.size(); ++i) {
        std::cerr << b.dims[i];
        if (i + 1 < b.dims.size()) std::cerr << ", ";
    }
    std::cerr << "]" << std::endl;
    
    // Handle broadcasting for 2D + 1D and 1D + 2D
    if (a.dims.size() == 2 && b.dims.size() == 1 && a.dims[1] == b.dims[0]) {
        // broadcast b to [a.dims[0], a.dims[1]]
        Tensor c(a.dims);
        for (int64_t i = 0; i < a.dims[0]; i++) {
            for (int64_t j = 0; j < a.dims[1]; j++) {
                c.data[i * a.dims[1] + j] = a.data[i * a.dims[1] + j] + b.data[j];
            }
        }
        return c;
    }
    if (a.dims.size() == 1 && b.dims.size() == 2 && a.dims[0] == b.dims[1]) {
        // broadcast a to [b.dims[0], b.dims[1]]
        Tensor c(b.dims);
        for (int64_t i = 0; i < b.dims[0]; i++) {
            for (int64_t j = 0; j < b.dims[1]; j++) {
                c.data[i * b.dims[1] + j] = a.data[j] + b.data[i * b.dims[1] + j];
            }
        }
        return c;
    }
    
    // Otherwise, require same dimensions
    assert(a.dims == b.dims);
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] + b.data[i];
    }
    return c;
}

// Element-wise subtraction: c = a - b
Tensor sub(const Tensor& a, const Tensor& b) {
    assert(a.dims == b.dims);
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] - b.data[i];
    }
    return c;
}

// Element-wise multiplication: c = a * b
Tensor mul(const Tensor& a, const Tensor& b) {
    assert(a.dims == b.dims);
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] * b.data[i];
    }
    return c;
}

// Element-wise division: c = a / b
Tensor div(const Tensor& a, const Tensor& b) {
    assert(a.dims == b.dims);
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] / b.data[i];
    }
    return c;
}

// Addition with scalar: c = a + scalar
Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] + scalar;
    }
    return c;
}

// Multiplication with scalar: c = a * scalar
Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor c(a.dims);
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] * scalar;
    }
    return c;
}

// Matrix multiplication: c = a * b (a: [M, K], b: [K, N] -> c: [M, N])
Tensor matmul(const Tensor& a, const Tensor& b) {
    assert(a.dims.size() == 2 && b.dims.size() == 2);
    int64_t M = a.dims[0];
    int64_t K = a.dims[1];
    int64_t N = b.dims[1];
    assert(b.dims[0] == K);
    
    Tensor c({M, N});
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += a.data[i * K + k] * b.data[k * N + j];
            }
            c.data[i * N + j] = sum;
        }
    }
    return c;
}

// Transpose last two dimensions (for 3D tensor: [B, T, C] -> [B, C, T])
Tensor transpose_last_two(const Tensor& a) {
    assert(a.dims.size() == 3);
    int64_t B = a.dims[0];
    int64_t T = a.dims[1];
    int64_t C = a.dims[2];
    
    Tensor result({B, C, T});
    for (int64_t b = 0; b < B; b++) {
        for (int64_t t = 0; t < T; t++) {
            for (int64_t c = 0; c < C; c++) {
                result.data[b * C * T + c * T + t] = a.data[b * T * C + t * C + c];
            }
        }
    }
    return result;
}

// Layer normalization: normalize over last dimension
Tensor layer_norm(const Tensor& x, float eps = 1e-5f) {
    assert(x.dims.size() >= 2);
    int64_t last_dim = x.dims.back();
    int64_t outer_size = 1;
    for (size_t i = 0; i < x.dims.size() - 1; i++) {
        outer_size *= x.dims[i];
    }
    
    Tensor y(x.dims);
    for (int64_t i = 0; i < outer_size; i++) {
        float mean = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            mean += x.data[idx];
        }
        mean /= last_dim;
        
        float variance = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            float diff = x.data[idx] - mean;
            variance += diff * diff;
        }
        variance /= last_dim;
        float inv_std = 1.0f / std::sqrt(variance + eps);
        
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            y.data[idx] = (x.data[idx] - mean) * inv_std;
        }
    }
    return y;
}

// Softmax over last dimension
Tensor softmax(const Tensor& x) {
    assert(x.dims.size() >= 2);
    int64_t last_dim = x.dims.back();
    int64_t outer_size = 1;
    for (size_t i = 0; i < x.dims.size() - 1; i++) {
        outer_size *= x.dims[i];
    }
    
    Tensor y(x.dims);
    for (int64_t i = 0; i < outer_size; i++) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            if (x.data[idx] > max_val) max_val = x.data[idx];
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            y.data[idx] = std::exp(x.data[idx] - max_val);
            sum_exp += y.data[idx];
        }
        
        // Normalize
        for (int64_t j = 0; j < last_dim; j++) {
            int64_t idx = i * last_dim + j;
            y.data[idx] /= sum_exp;
        }
    }
    return y;
}

// Cross entropy loss: input [N, C], target [N] (class indices)
float cross_entropy_loss(const Tensor& input, const Tensor& target) {
    assert(input.dims.size() == 2);
    assert(target.dims.size() == 1);
    assert(input.dims[0] == target.dims[0]);
    
    int64_t N = input.dims[0];
    int64_t C = input.dims[1];
    
    float loss = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        int64_t t = static_cast<int64_t>(target.data[i]);
        assert(t >= 0 && t < C);
        
        // Log softmax
        float max_val = -INFINITY;
        for (int64_t j = 0; j < C; j++) {
            float val = input.data[i * C + j];
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < C; j++) {
            sum_exp += std::exp(input.data[i * C + j] - max_val);
        }
        
        float log_prob = input.data[i * C + t] - max_val - std::log(sum_exp);
        loss -= log_prob;
    }
    return loss / N;
}

// GELU activation (approximate)
Tensor gelu(const Tensor& x) {
    Tensor y(x.dims);
    for (size_t i = 0; i < x.data.size(); i++) {
        float x_val = x.data[i];
        float cdf = 0.5f * (1.0f + std::erff(x_val / std::sqrt(2.0f)));
        y.data[i] = x_val * cdf;
    }
    return y;
}

// Embedding lookup: weight [vocab_size, embed_dim], indices [any_shape] -> output [indices.shape..., embed_dim]
Tensor embedding(const Tensor& weight, const Tensor& indices) {
    assert(weight.dims.size() == 2);
    int64_t vocab_size = weight.dims[0];
    int64_t embed_dim = weight.dims[1];
    
    // Flatten indices to 1D
    int64_t num_indices = 1;
    for (int64_t d : indices.dims) num_indices *= d;
    std::vector<int64_t> flat_indices_data(num_indices);
    for (int64_t i = 0; i < num_indices; i++) {
        flat_indices_data[i] = indices.data[i];
    }
    
    Tensor output({num_indices, embed_dim});
    for (int64_t i = 0; i < num_indices; i++) {
        int64_t idx = static_cast<int64_t>(flat_indices_data[i]);
        assert(idx >= 0 && idx < vocab_size);
        for (int64_t j = 0; j < embed_dim; j++) {
            output.data[i * embed_dim + j] = weight.data[idx * embed_dim + j];
        }
    }
    
    // Reshape output to [indices.dims..., embed_dim]
    std::vector<int64_t> output_dims = indices.dims;
    output_dims.push_back(embed_dim);
    output.reshape(output_dims);
    return output;
}

} // namespace ops

// Model configuration
struct GPTConfig {
    int vocab_size = 65;       // character-level: 65 unique chars in Shakespeare
    int block_size = 256;      // max sequence length (context window)
    int n_layer = 6;           // number of transformer blocks
    int n_head = 6;            // number of attention heads
    int n_embd = 384;          // embedding dimension
};

// Causal self-attention module
class CausalSelfAttention {
public:
    Tensor c_attn_weight;  // [n_embd, 3 * n_embd]
    Tensor c_attn_bias;    // [3 * n_embd]
    Tensor c_proj_weight;  // [n_embd, n_embd]
    Tensor c_proj_bias;    // [n_embd]
    
    int n_head;
    int n_embd;
    
    CausalSelfAttention(int n_embd, int n_head) : n_embd(n_embd), n_head(n_head) {
        // Initialize weights
        c_attn_weight = Tensor({n_embd, 3 * n_embd});
        c_attn_weight.randn(0.0f, 0.02f);
        c_attn_bias = Tensor({3 * n_embd});
        c_attn_bias.fill(0.0f);
        
        c_proj_weight = Tensor({n_embd, n_embd});
        c_proj_weight.randn(0.0f, 0.02f);
        c_proj_bias = Tensor({n_embd});
        c_proj_bias.fill(0.0f);
    }
    
    // Forward pass
    Tensor forward(const Tensor& x) {
        // x: [B, T, C]
        int64_t B = x.dims[0];
        int64_t T = x.dims[1];
        int64_t C = x.dims[2];
        assert(C == n_embd);
        
        // Reshape to [B*T, C] for efficiency
        Tensor x_flat({B * T, C});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < C; j++) {
                x_flat.data[i * C + j] = x.data[i * C + j];
            }
        }
        
        // QKV projection: [B*T, C] * [C, 3*C] -> [B*T, 3*C]
        Tensor qkv = ops::matmul(x_flat, c_attn_weight);
        qkv = ops::add(qkv, c_attn_bias);  // [B*T, 3*C]
        
        // Split into Q, K, V: each [B*T, C]
        Tensor Q({B * T, C});
        Tensor K({B * T, C});
        Tensor V({B * T, C});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < C; j++) {
                Q.data[i * C + j] = qkv.data[i * 3 * C + j];
                K.data[i * C + j] = qkv.data[i * 3 * C + C + j];
                V.data[i * C + j] = qkv.data[i * 3 * C + 2 * C + j];
            }
        }
        
        // Reshape for multi-head: [B*T, C] -> [B*T, n_head, head_dim]
        int64_t head_dim = C / n_head;
        Tensor Q_reshaped({B * T, n_head, head_dim});
        Tensor K_reshaped({B * T, n_head, head_dim});
        Tensor V_reshaped({B * T, n_head, head_dim});
        
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t h = 0; h < n_head; h++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t src_idx = i * C + h * head_dim + d;
                    int64_t dst_idx = i * n_head * head_dim + h * head_dim + d;
                    Q_reshaped.data[dst_idx] = Q.data[src_idx];
                    K_reshaped.data[dst_idx] = K.data[src_idx];
                    V_reshaped.data[dst_idx] = V.data[src_idx];
                }
            }
        }
        
        // Transpose to [B*T, n_head, head_dim] -> [n_head, B*T, head_dim] for batch matrix multiplication
        // We'll compute attention per head
        Tensor attn_output({B * T, C});
        
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (int64_t h = 0; h < n_head; h++) {
            // Extract head h: [B*T, head_dim]
            Tensor Q_h({B * T, head_dim});
            Tensor K_h({B * T, head_dim});
            Tensor V_h({B * T, head_dim});
            
            for (int64_t i = 0; i < B * T; i++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t src_idx = i * n_head * head_dim + h * head_dim + d;
                    Q_h.data[i * head_dim + d] = Q_reshaped.data[src_idx];
                    K_h.data[i * head_dim + d] = K_reshaped.data[src_idx];
                    V_h.data[i * head_dim + d] = V_reshaped.data[src_idx];
                }
            }
            
            // Actually, let's implement a proper transpose for 2D: [M, N] -> [N, M]
            auto transpose2d = [](const Tensor& t) {
                assert(t.dims.size() == 2);
                Tensor result({t.dims[1], t.dims[0]});
                for (int64_t i = 0; i < t.dims[0]; i++) {
                    for (int64_t j = 0; j < t.dims[1]; j++) {
                        result.data[j * t.dims[0] + i] = t.data[i * t.dims[1] + j];
                    }
                }
                return result;
            };
            
            Tensor K_h_T2 = transpose2d(K_h); // [head_dim, B*T]
            Tensor scores = ops::matmul(Q_h, K_h_T2); // [B*T, B*T]
            
            // Scale scores
            scores = ops::mul_scalar(scores, scale);
            
            // Apply causal mask: mask out future tokens
            for (int64_t i = 0; i < B * T; i++) {
                for (int64_t j = 0; j < B * T; j++) {
                    if (j > i) { // future token
                        scores.data[i * (B * T) + j] = -1e9f; // large negative
                    }
                }
            }
            
            // Softmax
            Tensor attn_weights = ops::softmax(scores); // [B*T, B*T]
            
            // Apply attention to V: attn_weights * V -> [B*T, head_dim]
            Tensor attn_h = ops::matmul(attn_weights, V_h); // [B*T, head_dim]
            
            // Store in output
            for (int64_t i = 0; i < B * T; i++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t dst_idx = i * n_head * head_dim + h * head_dim + d;
                    attn_output.data[dst_idx] = attn_h.data[i * head_dim + d];
                }
            }
        }
        
        // Output projection: [B*T, C] * [C, C] -> [B*T, C]
        Tensor proj_output = ops::matmul(attn_output, c_proj_weight);
        proj_output = ops::add(proj_output, c_proj_bias); // [B*T, C]
        
        // Reshape back to [B, T, C]
        Tensor output({B, T, C});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < C; j++) {
                output.data[i * C + j] = proj_output.data[i * C + j];
            }
        }
        
        return output;
    }
};

// MLP module
class MLP {
public:
    Tensor c_fc_weight;   // [n_embd, 4 * n_embd]
    Tensor c_fc_bias;     // [4 * n_embd]
    Tensor c_proj_weight; // [4 * n_embd, n_embd]
    Tensor c_proj_bias;   // [n_embd]
    
    int n_embd;
    
    MLP(int n_embd) : n_embd(n_embd) {
        // Initialize weights
        c_fc_weight = Tensor({n_embd, 4 * n_embd});
        c_fc_weight.randn(0.0f, 0.02f);
        c_fc_bias = Tensor({4 * n_embd});
        c_fc_bias.fill(0.0f);
        
        c_proj_weight = Tensor({4 * n_embd, n_embd});
        c_proj_weight.randn(0.0f, 0.02f);
        c_proj_bias = Tensor({n_embd});
        c_proj_bias.fill(0.0f);
    }
    
    // Forward pass
    Tensor forward(const Tensor& x) {
        // x: [B, T, C]
        int64_t B = x.dims[0];
        int64_t T = x.dims[1];
        int64_t C = x.dims[2];
        assert(C == n_embd);
        
        // Reshape to [B*T, C]
        Tensor x_flat({B * T, C});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < C; j++) {
                x_flat.data[i * C + j] = x.data[i * C + j];
            }
        }
        
        // FC layer: [B*T, C] * [C, 4*C] -> [B*T, 4*C]
        Tensor fc = ops::matmul(x_flat, c_fc_weight);
        fc = ops::add(fc, c_fc_bias); // [B*T, 4*C]
        
        // GELU activation
        Tensor gelu_out = ops::gelu(fc);
        
        // Projection: [B*T, 4*C] * [4*C, C] -> [B*T, C]
        Tensor proj = ops::matmul(gelu_out, c_proj_weight);
        proj = ops::add(proj, c_proj_bias); // [B*T, C]
        
        // Reshape back to [B, T, C]
        Tensor output({B, T, C});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < C; j++) {
                output.data[i * C + j] = proj.data[i * C + j];
            }
        }
        
        return output;
    }
};

// Transformer block
class Block {
public:
    Tensor ln_1_weight; // [n_embd]
    Tensor ln_1_bias;   // [n_embd]
    CausalSelfAttention attn;
    Tensor ln_2_weight; // [n_embd]
    Tensor ln_2_bias;   // [n_embd]
    MLP mlp;
    
    Block(int n_embd, int n_head) 
        : attn(n_embd, n_head), mlp(n_embd) {
        // LayerNorm weights
        ln_1_weight = Tensor({n_embd});
        ln_1_weight.fill(1.0f);
        ln_1_bias = Tensor({n_embd});
        ln_1_bias.fill(0.0f);
        
        ln_2_weight = Tensor({n_embd});
        ln_2_weight.fill(1.0f);
        ln_2_bias = Tensor({n_embd});
        ln_2_bias.fill(0.0f);
    }
    
    // Forward pass
    Tensor forward(const Tensor& x) {
        // x: [B, T, C]
        
        // Self-attention with residual connection
        Tensor ln_1 = ops::layer_norm(x);
        // Apply LayerNorm affine transformation
        Tensor ln_1_affine = ops::mul_scalar(ln_1_weight, 0.0f); // placeholder - we need to do per-element
        // Actually: y = weight * x + bias
        Tensor ln_1_affine_temp({ln_1.dims});
        for (size_t i = 0; i < ln_1.data.size(); i++) {
            ln_1_affine_temp.data[i] = ln_1_weight.data[i % ln_1_weight.dims[0]] * ln_1.data[i] + ln_1_bias.data[i % ln_1_bias.dims[0]];
        }
        Tensor attn_out = attn.forward(ln_1_affine_temp);
        Tensor attn_residual = ops::add(x, attn_out); // [B, T, C]
        
        // MLP with residual connection
        Tensor ln_2 = ops::layer_norm(attn_residual);
        Tensor ln_2_affine_temp({ln_2.dims});
        for (size_t i = 0; i < ln_2.data.size(); i++) {
            ln_2_affine_temp.data[i] = ln_2_weight.data[i % ln_2_weight.dims[0]] * ln_2.data[i] + ln_2_bias.data[i % ln_2_bias.dims[0]];
        }
        Tensor mlp_out = mlp.forward(ln_2_affine_temp);
        Tensor out = ops::add(attn_residual, mlp_out); // [B, T, C]
        
        return out;
    }
};

// GPT model
class GPT {
public:
    GPTConfig config;
    Tensor wte_weight;   // [vocab_size, n_embd]
    Tensor wpe_weight;   // [block_size, n_embd]
    std::vector<Block> blocks;
    Tensor ln_f_weight;  // [n_embd]
    Tensor ln_f_bias;    // [n_embd]
    
    GPT(GPTConfig config) : config(config) {
        // Token embedding
        wte_weight = Tensor({config.vocab_size, config.n_embd});
        wte_weight.randn(0.0f, 0.02f);
        
        // Position embedding
        wpe_weight = Tensor({config.block_size, config.n_embd});
        wpe_weight.randn(0.0f, 0.02f);
        
        // Transformer blocks
        for (int i = 0; i < config.n_layer; i++) {
            blocks.emplace_back(Block(config.n_embd, config.n_head));
        }
        
        // Final LayerNorm
        ln_f_weight = Tensor({config.n_embd});
        ln_f_weight.fill(1.0f);
        ln_f_bias = Tensor({config.n_embd});
        ln_f_bias.fill(0.0f);
    }
    
    // Forward pass
    // Returns logits and optionally loss if targets provided
    std::pair<Tensor, float> forward(const Tensor& idx, const Tensor* targets = nullptr) {
        // idx: [B, T] (token indices)
        int64_t B = idx.dims[0];
        int64_t T = idx.dims[1];
        assert(T <= config.block_size);
        
        // Token embeddings: [B, T, n_embd]
        Tensor tok_emb = ops::embedding(wte_weight, idx);
        // tok_emb is already [B, T, n_embd] because embedding preserves the shape of idx and appends n_embd
        
        // Position embeddings: [B, T, n_embd]
        Tensor pos_indices({B, T});
        for (int64_t b = 0; b < B; b++) {
            for (int64_t t = 0; t < T; t++) {
                pos_indices.data[b * T + t] = static_cast<float>(t);
            }
        }
        Tensor pos_emb = ops::embedding(wpe_weight, pos_indices); // [B, T, n_embd]
        
        // Embedding sum
        Tensor x = ops::add(tok_emb, pos_emb); // [B, T, n_embd]
        
        // Transformer blocks
        for (Block& block : blocks) {
            x = block.forward(x);
        }
        
        // Final LayerNorm
        Tensor ln_f = ops::layer_norm(x);
        Tensor ln_f_affine_temp({ln_f.dims});
        for (size_t i = 0; i < ln_f.data.size(); i++) {
            ln_f_affine_temp.data[i] = ln_f_weight.data[i % ln_f_weight.dims[0]] * ln_f.data[i] + ln_f_bias.data[i % ln_f_bias.dims[0]];
        }
        
        // Language model head: [B*T, n_embd] * [n_embd, vocab_size] -> [B*T, vocab_size]
        // Note: weight tying - we use wte_weight transposed as the output projection weight
        Tensor lm_head_input = ln_f_affine_temp;
        lm_head_input.reshape({B * T, config.n_embd});
        
        // Transpose wte_weight: [vocab_size, n_embd] -> [n_embd, vocab_size]
        Tensor wte_T = transpose2d(wte_weight); // [n_embd, vocab_size]
        Tensor logits = ops::matmul(lm_head_input, wte_T); // [B*T, vocab_size]
        
        // Calculate loss if targets provided
        float loss = 0.0f;
        if (targets != nullptr) {
            // Targets: [B, T] -> flatten to [B*T]
            Tensor targets_flat({B * T});
            for (int64_t i = 0; i < B * T; i++) {
                targets_flat.data[i] = targets->data[i];
            }
            loss = ops::cross_entropy_loss(logits, targets_flat);
        }
        
        // Reshape logits back to [B, T, vocab_size] if needed for generation
        Tensor logits_3d({B, T, config.vocab_size});
        for (int64_t i = 0; i < B * T; i++) {
            for (int64_t j = 0; j < config.vocab_size; j++) {
                logits_3d.data[i * config.vocab_size + j] = logits.data[i * config.vocab_size + j];
            }
        }
        
        return {logits_3d, loss};
    }
    
    // Helper: transpose 2D tensor
    static Tensor transpose2d(const Tensor& t) {
        assert(t.dims.size() == 2);
        Tensor result({t.dims[1], t.dims[0]});
        for (int64_t i = 0; i < t.dims[0]; i++) {
            for (int64_t j = 0; j < t.dims[1]; j++) {
                result.data[j * t.dims[0] + i] = t.data[i * t.dims[1] + j];
            }
        }
        return result;
    }
};

// AdamW optimizer
class AdamW {
public:
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t; // timestep
    
    // Moment and variance for each parameter
    std::vector<Tensor> m;
    std::vector<Tensor> v;
    
    AdamW(float lr = 1e-3f, float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), eps(epsilon), weight_decay(wd), t(0) {}
    
    // Initialize moment and variance tensors for parameters
    void init(const std::vector<Tensor>& params) {
        m.clear();
        v.clear();
        for (const Tensor& param : params) {
            m.push_back(Tensor(param.dims));
            m.back().fill(0.0f);
            v.push_back(Tensor(param.dims));
            v.back().fill(0.0f);
        }
    }
    
    // Update parameters
    void step(std::vector<Tensor>& params, const std::vector<Tensor>& grads) {
        t++;
        for (size_t i = 0; i < params.size(); i++) {
            Tensor& param = params[i];
            const Tensor& grad = grads[i];
            Tensor& mt = m[i];
            Tensor& vt = v[i];
            
            // Update biased first moment estimate
            for (size_t j = 0; j < param.data.size(); j++) {
                mt.data[j] = beta1 * mt.data[j] + (1 - beta1) * grad.data[j];
                vt.data[j] = beta2 * vt.data[j] + (1 - beta2) * grad.data[j] * grad.data[j];
            }
            
            // Compute bias-corrected first moment estimate
            Tensor mt_hat(mt.dims);
            Tensor vt_hat(vt.dims);
            for (size_t j = 0; j < param.data.size(); j++) {
                mt_hat.data[j] = mt.data[j] / (1 - std::pow(beta1, t));
                vt_hat.data[j] = vt.data[j] / (1 - std::pow(beta2, t));
            }
            
            // Update parameters
            for (size_t j = 0; j < param.data.size(); j++) {
                float denom = std::sqrt(vt_hat.data[j]) + eps;
                float update = learning_rate * (mt_hat.data[j] / denom + weight_decay * param.data[j]);
                param.data[j] -= update;
            }
        }
    }
};

// Utility functions for data loading and preprocessing
namespace utils {

// Load text file and create character mappings
std::tuple<std::vector<int>, std::unordered_map<char, int>, std::unordered_map<int, char>, int>
load_data(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();
    
    // Create vocabulary
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    int idx = 0;
    for (char c : text) {
        if (stoi.find(c) == stoi.end()) {
            stoi[c] = idx;
            itos[idx] = c;
            idx++;
        }
    }
    int vocab_size = idx;
    
    // Convert text to tokens
    std::vector<int> tokens;
    for (char c : text) {
        tokens.push_back(stoi[c]);
    }
    
    return {tokens, stoi, itos, vocab_size};
}

// Get batch of data
std::pair<std::vector<int>, std::vector<int>> get_batch(
    const std::vector<int>& tokens, 
    int block_size, 
    int batch_size,
    int start_idx) {
    
    // Ensure we don't go out of bounds
    int max_start = tokens.size() - block_size - 1;
    if (start_idx > max_start) start_idx = 0;
    
    std::vector<int> x(batch_size * block_size);
    std::vector<int> y(batch_size * block_size);
    
    for (int i = 0; i < batch_size; i++) {
        int idx = start_idx + i;
        if (idx > max_start) idx = 0; // wrap around
        
        for (int j = 0; j < block_size; j++) {
            x[i * block_size + j] = tokens[idx + j];
            y[i * block_size + j] = tokens[idx + j + 1];
        }
    }
    
    return {x, y};
}

} // namespace utils

#endif // MODEL_H