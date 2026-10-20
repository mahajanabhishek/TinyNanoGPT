#ifndef MODEL_H
#define MODEL_H

#include "logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Small OpenMP helper
// ============================================================================
namespace detail {

inline int thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

} // namespace detail

// ============================================================================
// Tensor
// ----------------------------------------------------------------------------
// This Tensor uses shared storage so copies are cheap.
// This is important because train.cpp copies model parameters into a vector:
//
//     params.push_back(model.wte_weight);
//
// With shared storage, optimizer.step(params, grads) updates the actual model
// weights instead of updating dead copies.
// ============================================================================
namespace gpt {

class TensorData {
public:
    TensorData()
        : storage_(std::make_shared<std::vector<float>>()) {}

    explicit TensorData(size_t n)
        : storage_(std::make_shared<std::vector<float>>(n, 0.0f)) {}

    TensorData(const TensorData&) = default;
    TensorData& operator=(const TensorData&) = default;

    float& operator[](size_t i) {
        return (*storage_)[i];
    }

    const float& operator[](size_t i) const {
        return (*storage_)[i];
    }

    float* data() {
        return storage_->data();
    }

    const float* data() const {
        return storage_->data();
    }

    size_t size() const {
        return storage_->size();
    }

    void resize(size_t n) {
        storage_->resize(n, 0.0f);
    }

    std::vector<float>::iterator begin() {
        return storage_->begin();
    }

    std::vector<float>::iterator end() {
        return storage_->end();
    }

    std::vector<float>::const_iterator begin() const {
        return storage_->begin();
    }

    std::vector<float>::const_iterator end() const {
        return storage_->end();
    }

private:
    std::shared_ptr<std::vector<float>> storage_;
};

class Tensor {
public:
    std::vector<int64_t> dims;
    TensorData data;

    Tensor() = default;

    Tensor(std::initializer_list<int64_t> dims_init)
        : dims(dims_init) {
        data.resize(static_cast<size_t>(elements()));
    }

    explicit Tensor(const std::vector<int64_t>& dims_init)
        : dims(dims_init) {
        data.resize(static_cast<size_t>(elements()));
    }

    int64_t elements() const {
        if (dims.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : dims) n *= d;
        return n;
    }

    Tensor clone() const {
        Tensor t;
        t.dims = dims;
        t.data.resize(static_cast<size_t>(elements()));
        std::copy(data.begin(), data.end(), t.data.begin());
        return t;
    }
};

} // namespace gpt

// Expose Tensor globally so existing code can use `Tensor` directly.
using gpt::Tensor;

// ============================================================================
// Implement logging helpers declared in logging.h
// ============================================================================
namespace gpt {
namespace log {

inline std::string fmt_shape(const ::gpt::Tensor& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.dims.size(); ++i) {
        if (i != 0) s += ", ";
        s += std::to_string(t.dims[i]);
    }
    s += "]";
    return s;
}

inline std::string fmt_tensor(const ::gpt::Tensor& t) {
    std::string s = fmt_shape(t) + " [";
    size_t n = std::min<size_t>(5, t.data.size());
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) s += ", ";
        s += std::to_string(t.data[i]);
    }
    if (t.data.size() > n) s += ", ...";
    s += "]";
    return s;
}

} // namespace log
} // namespace gpt

// ============================================================================
// Model structs
// ============================================================================
struct GPTConfig {
    int vocab_size = 0;
    int block_size = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_embd = 0;
};

struct Attention {
    Tensor c_attn_weight;   // [n_embd, 3 * n_embd]
    Tensor c_attn_bias;     // [3 * n_embd]
    Tensor c_proj_weight;   // [n_embd, n_embd]
    Tensor c_proj_bias;     // [n_embd]
};

struct MLP {
    Tensor c_fc_weight;     // [n_embd, 4 * n_embd]
    Tensor c_fc_bias;       // [4 * n_embd]
    Tensor c_proj_weight;   // [4 * n_embd, n_embd]
    Tensor c_proj_bias;     // [n_embd]
};

struct Block {
    Tensor ln_1_weight;
    Tensor ln_1_bias;
    Attention attn;
    Tensor ln_2_weight;
    Tensor ln_2_bias;
    MLP mlp;
};

// ============================================================================
// ops
// ============================================================================
namespace ops {

// ---------------------------------------------------------------------------
// Element-wise add
// ---------------------------------------------------------------------------
inline Tensor add(const Tensor& a, const Tensor& b) {
    GPT_LOG_ASSERT(a.dims == b.dims, "add: dimension mismatch");
    GPT_LOG_TRACE(
        "add: a.dims = %s, b.dims = %s",
        gpt::log::fmt_shape(a).c_str(),
        gpt::log::fmt_shape(b).c_str()
    );

    Tensor out(a.dims);
    int64_t n = static_cast<int64_t>(a.data.size());

    const float* ap = a.data.data();
    const float* bp = b.data.data();
    float* op = out.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        op[i] = ap[i] + bp[i];
    }

    return out;
}

inline void add_inplace(Tensor& a, const Tensor& b) {
    GPT_LOG_ASSERT(a.dims == b.dims, "add_inplace: dimension mismatch");

    int64_t n = static_cast<int64_t>(a.data.size());

    float* ap = a.data.data();
    const float* bp = b.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        ap[i] += bp[i];
    }
}

// ---------------------------------------------------------------------------
// Softmax over last dimension
// ---------------------------------------------------------------------------
inline Tensor softmax(const Tensor& x) {
    Tensor out(x.dims);

    if (x.dims.empty()) return out;

    int64_t last = x.dims.back();
    if (last <= 0) return out;

    int64_t rows = static_cast<int64_t>(x.data.size()) / last;

    const float* xp = x.data.data();
    float* op = out.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = xp + r * last;
        float* orow = op + r * last;

        float max_val = row[0];
        for (int64_t i = 1; i < last; ++i) {
            if (row[i] > max_val) max_val = row[i];
        }

        float sum = 0.0f;
        for (int64_t i = 0; i < last; ++i) {
            float e = std::exp(row[i] - max_val);
            orow[i] = e;
            sum += e;
        }

        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int64_t i = 0; i < last; ++i) {
            orow[i] *= inv_sum;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
inline void gelu_inplace(Tensor& x) {
    int64_t n = static_cast<int64_t>(x.data.size());
    float* p = x.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        p[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// ---------------------------------------------------------------------------
// LayerNorm over last dimension
// ---------------------------------------------------------------------------
inline Tensor layer_norm(
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& beta,
    float eps = 1e-5f
) {
    if (x.dims.empty()) return x.clone();

    int64_t C = x.dims.back();
    if (C <= 0) return Tensor(x.dims);

    int64_t rows = static_cast<int64_t>(x.data.size()) / C;

    Tensor out(x.dims);

    const float* xp = x.data.data();
    const float* gp = gamma.data.data();
    const float* bp = beta.data.data();
    float* op = out.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = xp + r * C;
        float* orow = op + r * C;

        double mean = 0.0;
        for (int64_t i = 0; i < C; ++i) {
            mean += row[i];
        }
        mean /= static_cast<double>(C);

        double var = 0.0;
        for (int64_t i = 0; i < C; ++i) {
            double d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<double>(C);

        float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);

        for (int64_t i = 0; i < C; ++i) {
            orow[i] = (row[i] - static_cast<float>(mean)) * inv * gp[i] + bp[i];
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Linear layer
// x: [..., K]
// weight: [K, N]
// bias: [N] optional
// output: [..., N]
// ---------------------------------------------------------------------------
inline Tensor linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias = nullptr
) {
    if (x.dims.empty()) return Tensor();

    GPT_LOG_ASSERT(weight.dims.size() == 2, "linear: weight must be 2D");
    GPT_LOG_ASSERT(x.dims.back() == weight.dims[0], "linear: input/weight mismatch");

    int64_t K = x.dims.back();
    int64_t M = static_cast<int64_t>(x.data.size()) / K;
    int64_t N = weight.dims[1];

    std::vector<int64_t> out_dims = x.dims;
    out_dims.back() = N;

    Tensor out(out_dims);

    const float* xp = x.data.data();
    const float* wp = weight.data.data();
    const float* bp = nullptr;

    if (bias != nullptr && bias->data.size() >= static_cast<size_t>(N)) {
        bp = bias->data.data();
    }

    float* op = out.data.data();

    #pragma omp parallel
    {
        std::vector<float> acc(static_cast<size_t>(N));

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < M; ++i) {
            if (bp != nullptr) {
                std::copy(bp, bp + N, acc.begin());
            } else {
                std::fill(acc.begin(), acc.end(), 0.0f);
            }

            const float* xrow = xp + i * K;

            for (int64_t k = 0; k < K; ++k) {
                float xv = xrow[k];
                const float* wrow = wp + k * N;

                for (int64_t j = 0; j < N; ++j) {
                    acc[j] += xv * wrow[j];
                }
            }

            std::copy(acc.begin(), acc.end(), op + i * N);
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Causal multi-head self-attention
// x: [B, T, C]
// ---------------------------------------------------------------------------
inline Tensor attention(
    const Tensor& x,
    const Attention& attn,
    int n_head
) {
    GPT_LOG_ASSERT(x.dims.size() == 3, "attention: expected [B, T, C]");

    int64_t B = x.dims[0];
    int64_t T = x.dims[1];
    int64_t C = x.dims[2];

    if (B <= 0 || T <= 0 || C <= 0 || n_head <= 0) {
        return x.clone();
    }

    int64_t H = static_cast<int64_t>(n_head);

    if (C % H != 0) {
        GPT_LOG_ASSERT(false, "attention: n_embd must be divisible by n_head");
        return x.clone();
    }

    int64_t D = C / H;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // QKV projection: [B, T, 3C]
    Tensor qkv = linear(x, attn.c_attn_weight, &attn.c_attn_bias);

    Tensor attn_out(x.dims);

    const float* qkv_p = qkv.data.data();
    float* out_p = attn_out.data.data();

    #pragma omp parallel
    {
        std::vector<float> scores(static_cast<size_t>(T));

        #pragma omp for collapse(2) schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {

                for (int64_t t = 0; t < T; ++t) {
                    int64_t base_t = (b * T + t) * 3 * C;
                    const float* q = qkv_p + base_t + h * D;

                    float max_score = -1e30f;

                    // Causal QK scores
                    for (int64_t j = 0; j <= t; ++j) {
                        int64_t base_j = (b * T + j) * 3 * C;
                        const float* k = qkv_p + base_j + C + h * D;

                        float score = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            score += q[d] * k[d];
                        }

                        score *= scale;
                        scores[j] = score;

                        if (score > max_score) {
                            max_score = score;
                        }
                    }

                    // Softmax
                    float sum = 0.0f;
                    for (int64_t j = 0; j <= t; ++j) {
                        float e = std::exp(scores[j] - max_score);
                        scores[j] = e;
                        sum += e;
                    }

                    float inv_sum = 1.0f / (sum + 1e-9f);

                    // Weighted sum of values
                    float* out_head = out_p + (b * T + t) * C + h * D;
                    std::fill(out_head, out_head + D, 0.0f);

                    for (int64_t j = 0; j <= t; ++j) {
                        float p = scores[j] * inv_sum;

                        int64_t base_j = (b * T + j) * 3 * C;
                        const float* v = qkv_p + base_j + 2 * C + h * D;

                        for (int64_t d = 0; d < D; ++d) {
                            out_head[d] += p * v[d];
                        }
                    }
                }

            }
        }
    }

    // Output projection
    Tensor proj = linear(attn_out, attn.c_proj_weight, &attn.c_proj_bias);
    return proj;
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------
inline Tensor mlp(const Tensor& x, const MLP& m) {
    Tensor h = linear(x, m.c_fc_weight, &m.c_fc_bias);
    gelu_inplace(h);
    Tensor out = linear(h, m.c_proj_weight, &m.c_proj_bias);
    return out;
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------
inline Tensor block_forward(
    const Tensor& x,
    const Block& block,
    int n_head
) {
    Tensor ln1 = layer_norm(x, block.ln_1_weight, block.ln_1_bias);
    Tensor attn = attention(ln1, block.attn, n_head);

    Tensor h = x.clone();
    add_inplace(h, attn);

    Tensor ln2 = layer_norm(h, block.ln_2_weight, block.ln_2_bias);
    Tensor ff = mlp(ln2, block.mlp);

    add_inplace(h, ff);
    return h;
}

// ---------------------------------------------------------------------------
// Token + positional embedding
// input: [B, T] or [T]
// output: [B, T, n_embd]
// ---------------------------------------------------------------------------
inline Tensor embed_tokens(
    const Tensor& input,
    const GPTConfig& config,
    const Tensor& wte,
    const Tensor& wpe
) {
    int64_t B = 1;
    int64_t T = 1;

    if (input.dims.size() == 1) {
        B = 1;
        T = input.dims[0];
    } else if (input.dims.size() >= 2) {
        B = input.dims[0];
        T = input.dims[1];
    }

    if (B <= 0 || T <= 0) return Tensor();

    if (config.vocab_size <= 0 || config.block_size <= 0 || config.n_embd <= 0) {
        return Tensor({B, T, std::max<int64_t>(0, static_cast<int64_t>(config.n_embd))});
    }

    int64_t C = static_cast<int64_t>(config.n_embd);
    Tensor out({B, T, C});

    int64_t N = B * T;
    int64_t input_n = static_cast<int64_t>(input.data.size());

    const float* ip = input.data.data();
    const float* wte_p = wte.data.data();
    const float* wpe_p = wpe.data.data();
    float* op = out.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        int64_t src_idx = (n < input_n) ? n : 0;

        int64_t token = static_cast<int64_t>(ip[src_idx]);
        if (token < 0) token = 0;
        if (token >= config.vocab_size) token = config.vocab_size - 1;

        int64_t pos = n % T;
        if (pos >= config.block_size) pos = config.block_size - 1;

        const float* token_emb = wte_p + token * C;
        const float* pos_emb = wpe_p + pos * C;
        float* o = op + n * C;

        for (int64_t c = 0; c < C; ++c) {
            o[c] = token_emb[c] + pos_emb[c];
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// LM head: hidden [B,T,C] -> logits [B,T,V]
// Uses weight tying with wte: logits = hidden * wte^T
// ---------------------------------------------------------------------------
inline Tensor lm_head(
    const Tensor& hidden,
    const Tensor& wte,
    const std::vector<int64_t>& batch_dims
) {
    if (hidden.dims.empty()) return Tensor();

    int64_t C = hidden.dims.back();
    int64_t N = static_cast<int64_t>(hidden.data.size()) / C;
    int64_t V = wte.dims[0];

    std::vector<int64_t> out_dims = batch_dims;
    out_dims.push_back(V);

    Tensor out(out_dims);

    if (C <= 0 || V <= 0 || N <= 0) return out;

    const float* hp = hidden.data.data();
    const float* wp = wte.data.data();
    float* op = out.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const float* hrow = hp + i * C;
        float* orow = op + i * V;

        for (int64_t v = 0; v < V; ++v) {
            const float* wrow = wp + v * C;
            float sum = 0.0f;

            for (int64_t c = 0; c < C; ++c) {
                sum += hrow[c] * wrow[c];
            }

            orow[v] = sum;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Cross-entropy loss directly from hidden states and tied LM head.
// This avoids materializing a huge logits tensor during training.
// ---------------------------------------------------------------------------
inline float cross_entropy_from_hidden(
    const Tensor& hidden,
    const Tensor& target,
    const Tensor& wte
) {
    if (hidden.dims.empty()) return 0.0f;

    int64_t C = hidden.dims.back();
    int64_t N = static_cast<int64_t>(hidden.data.size()) / C;
    int64_t V = wte.dims[0];

    if (C <= 0 || V <= 0 || N <= 0) return 0.0f;
    if (target.data.size() == 0) return 0.0f;

    const float* hp = hidden.data.data();
    const float* wp = wte.data.data();
    const float* tp = target.data.data();
    int64_t target_n = static_cast<int64_t>(target.data.size());

    double total_loss = 0.0;

    #pragma omp parallel reduction(+:total_loss)
    {
        std::vector<float> logits(static_cast<size_t>(V));

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            const float* hrow = hp + i * C;

            float max_logit = -1e30f;

            for (int64_t v = 0; v < V; ++v) {
                const float* wrow = wp + v * C;
                float sum = 0.0f;

                for (int64_t c = 0; c < C; ++c) {
                    sum += hrow[c] * wrow[c];
                }

                logits[v] = sum;
                if (sum > max_logit) max_logit = sum;
            }

            int64_t target_idx = (i < target_n) ? i : (i % target_n);
            int64_t target_token = static_cast<int64_t>(tp[target_idx]);

            if (target_token < 0) target_token = 0;
            if (target_token >= V) target_token = V - 1;

            float target_logit = logits[target_token];

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                sum_exp += std::exp(logits[v] - max_logit);
            }

            // Stable cross entropy:
            // loss = -target_logit + log(sum(exp(logits)))
            //      = -target_logit + max_logit + log(sum(exp(logits - max_logit)))
            total_loss += static_cast<double>(
                std::log(sum_exp + 1e-12f) + max_logit - target_logit
            );
        }
    }

    return static_cast<float>(total_loss / static_cast<double>(N));
}

} // namespace ops

// ============================================================================
// Initialization helpers
// ============================================================================
namespace init_detail {

inline void fill(Tensor& t, float value) {
    int64_t n = static_cast<int64_t>(t.data.size());
    float* p = t.data.data();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        p[i] = value;
    }
}

inline void normal(Tensor& t, float mean, float stddev, int seed) {
    int64_t n = static_cast<int64_t>(t.data.size());
    if (n == 0) return;

    if (stddev <= 0.0f) {
        fill(t, mean);
        return;
    }

    float* p = t.data.data();

    #pragma omp parallel
    {
        std::mt19937 gen(static_cast<unsigned int>(
            seed + detail::thread_num() * 1000003
        ));

        std::normal_distribution<float> dist(mean, stddev);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            p[i] = dist(gen);
        }
    }
}

} // namespace init_detail

// ============================================================================
// AdamW optimizer
// ============================================================================
class AdamW {
public:
    float learning_rate;
    int t;

    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;

    std::vector<Tensor> m;
    std::vector<Tensor> v;

    AdamW(float lr = 1e-3f, float wd = 0.01f)
        : learning_rate(lr),
          t(0),
          beta1(0.9f),
          beta2(0.999f),
          epsilon(1e-8f),
          weight_decay(wd) {}

    void init(const std::vector<Tensor>& params) {
        m.resize(params.size());
        v.resize(params.size());

        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = Tensor(params[i].dims);
            v[i] = Tensor(params[i].dims);
            init_detail::fill(m[i], 0.0f);
            init_detail::fill(v[i], 0.0f);
        }
    }

    void step(
        std::vector<Tensor>& params,
        const std::vector<Tensor>& grads
    ) {
        if (params.size() != grads.size()) return;

        if (m.size() != params.size() || v.size() != params.size()) {
            init(params);
        }

        ++t;

        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

        float decay_factor = 1.0f - learning_rate * weight_decay;

        for (size_t p = 0; p < params.size(); ++p) {
            int64_t n = static_cast<int64_t>(params[p].data.size());

            if (n == 0) continue;
            if (grads[p].data.size() != static_cast<size_t>(n)) continue;
            if (m[p].data.size() != static_cast<size_t>(n)) continue;
            if (v[p].data.size() != static_cast<size_t>(n)) continue;

            float* theta = params[p].data.data();
            const float* g = grads[p].data.data();
            float* mp = m[p].data.data();
            float* vp = v[p].data.data();

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                float gi = g[i];

                float mi = beta1 * mp[i] + (1.0f - beta1) * gi;
                float vi = beta2 * vp[i] + (1.0f - beta2) * gi * gi;

                mp[i] = mi;
                vp[i] = vi;

                float m_hat = mi / bc1;
                float v_hat = vi / bc2;

                float adam_update = m_hat / (std::sqrt(v_hat) + epsilon);

                theta[i] = theta[i] * decay_factor - learning_rate * adam_update;
            }
        }
    }
};

// ============================================================================
// GPT model
// ============================================================================
class GPT {
public:
    GPTConfig config;

    Tensor wte_weight;  // [vocab_size, n_embd]
    Tensor wpe_weight;  // [block_size, n_embd]

    std::vector<Block> blocks;

    Tensor ln_f_weight; // [n_embd]
    Tensor ln_f_bias;   // [n_embd]

    GPT() = default;

    explicit GPT(const GPTConfig& cfg) {
        init(cfg);
    }

    void init(const GPTConfig& cfg) {
        config = cfg;

        if (config.vocab_size <= 0) config.vocab_size = 1;
        if (config.block_size <= 0) config.block_size = 1;
        if (config.n_layer <= 0) config.n_layer = 1;
        if (config.n_head <= 0) config.n_head = 1;
        if (config.n_embd <= 0) config.n_embd = 1;

        int64_t V = static_cast<int64_t>(config.vocab_size);
        int64_t T = static_cast<int64_t>(config.block_size);
        int64_t C = static_cast<int64_t>(config.n_embd);

        wte_weight = Tensor({V, C});
        wpe_weight = Tensor({T, C});

        init_detail::normal(wte_weight, 0.0f, 0.02f, 1001);
        init_detail::normal(wpe_weight, 0.0f, 0.02f, 1002);

        blocks.resize(static_cast<size_t>(config.n_layer));

        for (int l = 0; l < config.n_layer; ++l) {
            Block& b = blocks[l];

            int seed_base = 2000 + l * 100;

            b.ln_1_weight = Tensor({C});
            b.ln_1_bias = Tensor({C});
            init_detail::fill(b.ln_1_weight, 1.0f);
            init_detail::fill(b.ln_1_bias, 0.0f);

            b.attn.c_attn_weight = Tensor({C, 3 * C});
            b.attn.c_attn_bias = Tensor({3 * C});
            init_detail::normal(b.attn.c_attn_weight, 0.0f, 0.02f, seed_base + 1);
            init_detail::fill(b.attn.c_attn_bias, 0.0f);

            b.attn.c_proj_weight = Tensor({C, C});
            b.attn.c_proj_bias = Tensor({C});
            init_detail::normal(b.attn.c_proj_weight, 0.0f, 0.02f, seed_base + 2);
            init_detail::fill(b.attn.c_proj_bias, 0.0f);

            b.ln_2_weight = Tensor({C});
            b.ln_2_bias = Tensor({C});
            init_detail::fill(b.ln_2_weight, 1.0f);
            init_detail::fill(b.ln_2_bias, 0.0f);

            b.mlp.c_fc_weight = Tensor({C, 4 * C});
            b.mlp.c_fc_bias = Tensor({4 * C});
            init_detail::normal(b.mlp.c_fc_weight, 0.0f, 0.02f, seed_base + 3);
            init_detail::fill(b.mlp.c_fc_bias, 0.0f);

            b.mlp.c_proj_weight = Tensor({4 * C, C});
            b.mlp.c_proj_bias = Tensor({C});
            init_detail::normal(b.mlp.c_proj_weight, 0.0f, 0.02f, seed_base + 4);
            init_detail::fill(b.mlp.c_proj_bias, 0.0f);
        }

        ln_f_weight = Tensor({C});
        ln_f_bias = Tensor({C});
        init_detail::fill(ln_f_weight, 1.0f);
        init_detail::fill(ln_f_bias, 0.0f);
    }

    // ------------------------------------------------------------------------
    // forward()
    //
    // input:
    //   - training: [batch_size, block_size]
    //   - generation: [context_len]
    //
    // target:
    //   - nullptr: return logits for generation
    //   - non-null: return loss for training and skip materializing logits
    // ------------------------------------------------------------------------
    std::pair<Tensor, float> forward(
        const Tensor& input,
        const Tensor* target = nullptr
    ) {
        if (config.n_embd <= 0) {
            return {Tensor(), 0.0f};
        }

        int64_t B = 1;
        int64_t T = 1;

        if (input.dims.size() == 1) {
            B = 1;
            T = input.dims[0];
        } else if (input.dims.size() >= 2) {
            B = input.dims[0];
            T = input.dims[1];
        }

        if (B <= 0 || T <= 0) {
            return {Tensor(), 0.0f};
        }

        Tensor hidden = ops::embed_tokens(input, config, wte_weight, wpe_weight);

        for (const Block& block : blocks) {
            hidden = ops::block_forward(hidden, block, config.n_head);
        }

        hidden = ops::layer_norm(hidden, ln_f_weight, ln_f_bias);

        if (target == nullptr) {
            Tensor logits = ops::lm_head(hidden, wte_weight, {B, T});
            return {logits, 0.0f};
        }

        float loss = ops::cross_entropy_from_hidden(hidden, *target, wte_weight);
        return {Tensor(), loss};
    }
};

// ============================================================================
// utils
// ============================================================================
namespace utils {

inline std::tuple<
    std::vector<int>,
    std::unordered_map<char, int>,
    std::unordered_map<int, char>,
    int
> load_data(const std::string& path) {
    std::string text;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: could not open data file: " << path << std::endl;
        std::cerr << "Using built-in fallback text." << std::endl;

        text =
            "First Citizen:\n"
            "Before we proceed any further, hear me speak.\n"
            "\n"
            "All:\n"
            "Speak, speak.\n"
            "\n"
            "First Citizen:\n"
            "You are all resolved rather to die than to famish?\n";
    } else {
        text.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    }

    if (text.empty()) {
        text = "hello world\n";
    }

    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    std::vector<int> tokens;

    stoi.reserve(text.size());
    itos.reserve(text.size());
    tokens.reserve(text.size());

    for (char c : text) {
        auto it = stoi.find(c);
        if (it == stoi.end()) {
            int idx = static_cast<int>(stoi.size());
            stoi[c] = idx;
            itos[idx] = c;
            tokens.push_back(idx);
        } else {
            tokens.push_back(it->second);
        }
    }

    int vocab_size = static_cast<int>(stoi.size());

    return {
        std::move(tokens),
        std::move(stoi),
        std::move(itos),
        vocab_size
    };
}

inline std::pair<std::vector<int>, std::vector<int>> get_batch(
    const std::vector<int>& tokens,
    int block_size,
    int batch_size,
    int token_idx
) {
    if (batch_size <= 0 || block_size <= 0) {
        return {{}, {}};
    }

    size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(block_size);

    std::vector<int> x(total, 0);
    std::vector<int> y(total, 0);

    int n = static_cast<int>(tokens.size());
    if (n == 0) {
        return {x, y};
    }

    int range = std::max(1, n - block_size - 1);

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; ++b) {
        int raw = token_idx + b;
        int start = raw % range;
        if (start < 0) start += range;

        for (int t = 0; t < block_size; ++t) {
            int x_idx = (start + t) % n;
            int y_idx = (start + t + 1) % n;

            size_t flat = static_cast<size_t>(b) * static_cast<size_t>(block_size) +
                          static_cast<size_t>(t);

            x[flat] = tokens[x_idx];
            y[flat] = tokens[y_idx];
        }
    }

    return {std::move(x), std::move(y)};
}

} // namespace utils

#endif // MODEL_H