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
#include <limits>
#include <utility>

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
Tensor layer_norm(const Tensor& x, float eps = 1e-5f, std::vector<float>* inverse_std = nullptr) {
    assert(x.dims.size() >= 2);
    int64_t last_dim = x.dims.back();
    int64_t outer_size = 1;
    for (size_t i = 0; i < x.dims.size() - 1; i++) {
        outer_size *= x.dims[i];
    }
    
    if (inverse_std) inverse_std->resize(outer_size);
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
        if (inverse_std) (*inverse_std)[i] = inv_std;
        
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

using Gradients = std::unordered_map<const Tensor*, Tensor>;

inline Tensor& gradient_for(Gradients& gradients, const Tensor& parameter) {
    return gradients.try_emplace(&parameter, parameter.dims).first->second;
}

namespace ops {

inline Tensor transpose2d(const Tensor& input) {
    const auto rows = input.dims[0], cols = input.dims[1];
    Tensor output({cols, rows});
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            output.data[col * rows + row] = input.data[row * cols + col];
        }
    }
    return output;
}

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return add(matmul(input, weight), bias);
}

inline Tensor linear_backward(const Tensor& input, const Tensor& weight, const Tensor& bias,
                              const Tensor& upstream, Gradients& gradients) {
    gradient_for(gradients, weight) = matmul(transpose2d(input), upstream);
    Tensor& bias_gradient = gradient_for(gradients, bias);
    const auto cols = bias.dims[0];
    for (size_t i = 0; i < upstream.data.size(); ++i) {
        bias_gradient.data[i % cols] += upstream.data[i];
    }
    return matmul(upstream, transpose2d(weight));
}

struct LayerNormCache {
    Tensor normalized;
    std::vector<float> inverse_std;
};

inline Tensor affine_layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                LayerNormCache* cache) {
    Tensor normalized = layer_norm(input, 1e-5f, cache ? &cache->inverse_std : nullptr);
    Tensor output(input.dims);
    const auto cols = weight.dims[0];
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] = normalized.data[i] * weight.data[i % cols] + bias.data[i % cols];
    }
    if (cache) cache->normalized = std::move(normalized);
    return output;
}

inline Tensor layer_norm_backward(const LayerNormCache& cache, const Tensor& weight, const Tensor& bias,
                                  const Tensor& upstream, Gradients& gradients) {
    Tensor& weight_gradient = gradient_for(gradients, weight);
    Tensor& bias_gradient = gradient_for(gradients, bias);
    Tensor input_gradient(upstream.dims);
    const auto cols = weight.dims[0];
    for (size_t row = 0; row < cache.inverse_std.size(); ++row) {
        float sum = 0.0f, normalized_sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            const auto i = row * cols + col;
            const float scaled = upstream.data[i] * weight.data[col];
            sum += scaled;
            normalized_sum += scaled * cache.normalized.data[i];
            weight_gradient.data[col] += upstream.data[i] * cache.normalized.data[i];
            bias_gradient.data[col] += upstream.data[i];
        }
        for (int64_t col = 0; col < cols; ++col) {
            const auto i = row * cols + col;
            input_gradient.data[i] = cache.inverse_std[row] *
                (upstream.data[i] * weight.data[col] -
                 (sum + cache.normalized.data[i] * normalized_sum) / cols);
        }
    }
    return input_gradient;
}

} // namespace ops

class CausalSelfAttention {
public:
    struct Cache {
        Tensor input;
        Tensor qkv;
        Tensor probabilities;
        Tensor context;
    };

    Tensor c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias;
    int n_head, n_embd;

    CausalSelfAttention(int n_embd, int n_head)
        : c_attn_weight({n_embd, 3 * n_embd}), c_attn_bias({3 * n_embd}),
          c_proj_weight({n_embd, n_embd}), c_proj_bias({n_embd}),
          n_head(n_head), n_embd(n_embd) {
        c_attn_weight.randn();
        c_proj_weight.randn();
    }

    Tensor forward(const Tensor& input, int64_t batch, int64_t time, Cache* cache) const {
        const int64_t channels = n_embd, head_size = channels / n_head;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        Tensor qkv = ops::linear(input, c_attn_weight, c_attn_bias);
        Tensor context({batch * time, channels});
        Tensor probabilities;
        if (cache) probabilities = Tensor({batch, n_head, time, time});
        std::vector<float> row(time);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_head; ++h) {
                const auto head = h * head_size;
                for (int64_t t = 0; t < time; ++t) {
                    const auto query = (b * time + t) * 3 * channels + head;
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (int64_t s = 0; s <= t; ++s) {
                        const auto key = (b * time + s) * 3 * channels + channels + head;
                        float score = 0.0f;
                        for (int64_t d = 0; d < head_size; ++d) {
                            score += qkv.data[query + d] * qkv.data[key + d];
                        }
                        row[s] = score * scale;
                        max_score = std::max(max_score, row[s]);
                    }

                    float sum = 0.0f;
                    for (int64_t s = 0; s <= t; ++s) {
                        row[s] = std::exp(row[s] - max_score);
                        sum += row[s];
                    }
                    const auto out = (b * time + t) * channels + head;
                    const auto probability_row = ((b * n_head + h) * time + t) * time;
                    for (int64_t s = 0; s <= t; ++s) {
                        const float probability = row[s] / sum;
                        if (cache) probabilities.data[probability_row + s] = probability;
                        const auto value = (b * time + s) * 3 * channels + 2 * channels + head;
                        for (int64_t d = 0; d < head_size; ++d) {
                            context.data[out + d] += probability * qkv.data[value + d];
                        }
                    }
                }
            }
        }
        Tensor output = ops::linear(context, c_proj_weight, c_proj_bias);
        if (cache) *cache = {input, std::move(qkv), std::move(probabilities), std::move(context)};
        return output;
    }

    Tensor backward(const Cache& cache, const Tensor& upstream, Gradients& gradients) const {
        const auto batch = cache.probabilities.dims[0], time = cache.probabilities.dims[2];
        const int64_t channels = n_embd, head_size = channels / n_head;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        Tensor context_gradient = ops::linear_backward(cache.context, c_proj_weight, c_proj_bias,
                                                       upstream, gradients);
        Tensor qkv_gradient(cache.qkv.dims);
        std::vector<float> value_dots(time);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_head; ++h) {
                const auto head = h * head_size;
                for (int64_t t = 0; t < time; ++t) {
                    const auto out = (b * time + t) * channels + head;
                    const auto query = (b * time + t) * 3 * channels + head;
                    const auto probability_row = ((b * n_head + h) * time + t) * time;
                    float mean_dot = 0.0f;
                    for (int64_t s = 0; s <= t; ++s) {
                        const auto value = (b * time + s) * 3 * channels + 2 * channels + head;
                        const float probability = cache.probabilities.data[probability_row + s];
                        float dot = 0.0f;
                        for (int64_t d = 0; d < head_size; ++d) {
                            const float delta = context_gradient.data[out + d];
                            dot += delta * cache.qkv.data[value + d];
                            qkv_gradient.data[value + d] += probability * delta;
                        }
                        value_dots[s] = dot;
                        mean_dot += probability * dot;
                    }

                    // Apply the softmax Jacobian to the probability gradients.
                    for (int64_t s = 0; s <= t; ++s) {
                        const auto key = (b * time + s) * 3 * channels + channels + head;
                        const float delta = scale * cache.probabilities.data[probability_row + s] *
                                            (value_dots[s] - mean_dot);
                        for (int64_t d = 0; d < head_size; ++d) {
                            qkv_gradient.data[query + d] += delta * cache.qkv.data[key + d];
                            qkv_gradient.data[key + d] += delta * cache.qkv.data[query + d];
                        }
                    }
                }
            }
        }
        return ops::linear_backward(cache.input, c_attn_weight, c_attn_bias, qkv_gradient, gradients);
    }
};

class MLP {
public:
    struct Cache {
        Tensor input;
        Tensor pre_activation;
        Tensor activation;
    };

    Tensor c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias;

    MLP(int channels)
        : c_fc_weight({channels, 4 * channels}), c_fc_bias({4 * channels}),
          c_proj_weight({4 * channels, channels}), c_proj_bias({channels}) {
        c_fc_weight.randn();
        c_proj_weight.randn();
    }

    Tensor forward(const Tensor& input, Cache* cache) const {
        Tensor pre_activation = ops::linear(input, c_fc_weight, c_fc_bias);
        Tensor activation = ops::gelu(pre_activation);
        Tensor output = ops::linear(activation, c_proj_weight, c_proj_bias);
        if (cache) *cache = {input, std::move(pre_activation), std::move(activation)};
        return output;
    }

    Tensor backward(const Cache& cache, const Tensor& upstream, Gradients& gradients) const {
        Tensor delta = ops::linear_backward(cache.activation, c_proj_weight, c_proj_bias,
                                            upstream, gradients);
        constexpr float inverse_sqrt_2pi = 0.3989422804f;
        for (size_t i = 0; i < delta.data.size(); ++i) {
            const float x = cache.pre_activation.data[i];
            const float cdf = 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
            delta.data[i] *= cdf + x * inverse_sqrt_2pi * std::exp(-0.5f * x * x);
        }
        return ops::linear_backward(cache.input, c_fc_weight, c_fc_bias, delta, gradients);
    }
};

class Block {
public:
    struct Cache {
        ops::LayerNormCache ln_1, ln_2;
        CausalSelfAttention::Cache attention;
        MLP::Cache mlp;
    };

    Tensor ln_1_weight, ln_1_bias;
    CausalSelfAttention attn;
    Tensor ln_2_weight, ln_2_bias;
    MLP mlp;

    Block(int channels, int heads)
        : ln_1_weight({channels}), ln_1_bias({channels}), attn(channels, heads),
          ln_2_weight({channels}), ln_2_bias({channels}), mlp(channels) {
        ln_1_weight.fill(1.0f);
        ln_2_weight.fill(1.0f);
    }

    Tensor forward(const Tensor& input, int64_t batch, int64_t time, Cache* cache) const {
        Tensor normalized = ops::affine_layer_norm(input, ln_1_weight, ln_1_bias,
                                                   cache ? &cache->ln_1 : nullptr);
        Tensor residual = ops::add(input, attn.forward(normalized, batch, time,
                                                       cache ? &cache->attention : nullptr));
        normalized = ops::affine_layer_norm(residual, ln_2_weight, ln_2_bias,
                                            cache ? &cache->ln_2 : nullptr);
        return ops::add(residual, mlp.forward(normalized, cache ? &cache->mlp : nullptr));
    }

    Tensor backward(const Cache& cache, const Tensor& upstream, Gradients& gradients) const {
        Tensor delta = mlp.backward(cache.mlp, upstream, gradients);
        Tensor residual = ops::add(upstream, ops::layer_norm_backward(cache.ln_2, ln_2_weight, ln_2_bias,
                                                                     delta, gradients));
        delta = attn.backward(cache.attention, residual, gradients);
        return ops::add(residual, ops::layer_norm_backward(cache.ln_1, ln_1_weight, ln_1_bias,
                                                            delta, gradients));
    }
};

class GPT {
public:
    GPTConfig config;
    Tensor wte_weight, wpe_weight;
    std::vector<Block> blocks;
    Tensor ln_f_weight, ln_f_bias;

    GPT(GPTConfig config) : config(config) {
        wte_weight = Tensor({config.vocab_size, config.n_embd});
        wte_weight.randn();
        wpe_weight = Tensor({config.block_size, config.n_embd});
        wpe_weight.randn();
        blocks.reserve(config.n_layer);
        for (int i = 0; i < config.n_layer; ++i) {
            blocks.emplace_back(config.n_embd, config.n_head);
        }
        ln_f_weight = Tensor({config.n_embd});
        ln_f_weight.fill(1.0f);
        ln_f_bias = Tensor({config.n_embd});
    }

    std::pair<Tensor, float> forward(const Tensor& idx, const Tensor* targets = nullptr) const {
        validate_input(idx, targets);
        Tensor logits = forward_logits(idx, nullptr);
        float loss = 0.0f;
        if (targets) {
            Tensor flat_targets = *targets;
            flat_targets.reshape({flat_targets.size()});
            loss = ops::cross_entropy_loss(logits, flat_targets);
        }
        logits.reshape({idx.dims[0], idx.dims[1], config.vocab_size});
        return {std::move(logits), loss};
    }

    float loss_and_gradients(const Tensor& idx, const Tensor& targets, Gradients& gradients) const {
        validate_input(idx, &targets);
        gradients.clear();
        Cache cache;
        Tensor logits = forward_logits(idx, &cache);
        Tensor flat_targets = targets;
        flat_targets.reshape({flat_targets.size()});
        const float loss = ops::cross_entropy_loss(logits, flat_targets);
        Tensor delta = ops::softmax(logits);
        const auto rows = logits.dims[0], time = idx.dims[1];
        for (int64_t row = 0; row < rows; ++row) {
            const auto target = static_cast<int64_t>(targets.data[row]);
            delta.data[row * config.vocab_size + target] -= 1.0f;
        }
        for (float& value : delta.data) value /= static_cast<float>(rows);

        Tensor& token_gradient = gradient_for(gradients, wte_weight);
        token_gradient = ops::matmul(ops::transpose2d(delta), cache.head_input);
        delta = ops::matmul(delta, wte_weight);
        delta = ops::layer_norm_backward(cache.final_norm, ln_f_weight, ln_f_bias, delta, gradients);
        for (size_t i = blocks.size(); i > 0; --i) {
            delta = blocks[i - 1].backward(cache.blocks[i - 1], delta, gradients);
        }

        // The token embedding receives gradients from both lookup and the tied output head.
        Tensor& position_gradient = gradient_for(gradients, wpe_weight);
        const int64_t channels = config.n_embd;
        for (int64_t row = 0; row < rows; ++row) {
            const auto token = static_cast<int64_t>(idx.data[row]);
            for (int64_t c = 0; c < channels; ++c) {
                const float value = delta.data[row * channels + c];
                token_gradient.data[token * channels + c] += value;
                position_gradient.data[(row % time) * channels + c] += value;
            }
        }
        return loss;
    }

private:
    struct Cache {
        std::vector<Block::Cache> blocks;
        ops::LayerNormCache final_norm;
        Tensor head_input;
    };

    void validate_input(const Tensor& idx, const Tensor* targets) const {
        if (idx.dims.size() != 2 || idx.dims[0] < 1 || idx.dims[1] < 1 ||
            idx.dims[1] > config.block_size || (targets && targets->dims != idx.dims)) {
            throw std::invalid_argument("Expected [batch, time] tokens with time within the context window");
        }
    }

    Tensor forward_logits(const Tensor& idx, Cache* cache) const {
        const auto batch = idx.dims[0], time = idx.dims[1];
        Tensor hidden = ops::embedding(wte_weight, idx);
        hidden.reshape({batch * time, config.n_embd});
        for (int64_t row = 0; row < batch * time; ++row) {
            const auto pos = row % time;
            for (int c = 0; c < config.n_embd; ++c) {
                hidden.data[row * config.n_embd + c] += wpe_weight.data[pos * config.n_embd + c];
            }
        }
        if (cache) cache->blocks.resize(blocks.size());
        for (size_t i = 0; i < blocks.size(); ++i) {
            hidden = blocks[i].forward(hidden, batch, time, cache ? &cache->blocks[i] : nullptr);
        }
        Tensor head_input = ops::affine_layer_norm(hidden, ln_f_weight, ln_f_bias,
                                                    cache ? &cache->final_norm : nullptr);
        Tensor logits = ops::matmul(head_input, ops::transpose2d(wte_weight));
        if (cache) cache->head_input = std::move(head_input);
        return logits;
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
    void init(const std::vector<Tensor*>& params) {
        m.clear();
        v.clear();
        for (const Tensor* param : params) {
            m.push_back(Tensor(param->dims));
            m.back().fill(0.0f);
            v.push_back(Tensor(param->dims));
            v.back().fill(0.0f);
        }
    }
    
    // Update parameters
    void step(const std::vector<Tensor*>& params, const Gradients& grads) {
        t++;
        for (size_t i = 0; i < params.size(); i++) {
            Tensor& param = *params[i];
            const Tensor& grad = grads.at(params[i]);
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
