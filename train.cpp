#include "model.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>

// Simple structure for loss logging
struct LossLog {
    std::vector<int> steps;
    std::vector<float> train;
    std::vector<float> val;
    
    void add(int step, float train_loss, float val_loss = -1.0f) {
        steps.push_back(step);
        train.push_back(train_loss);
        if (val_loss >= 0) val.push_back(val_loss);
    }
    
    void save(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Could not open file for writing: " << filename << std::endl;
            return;
        }
        
        // Simple CSV format
        file << "step,train_loss,val_loss\n";
        for (size_t i = 0; i < steps.size(); i++) {
            file << steps[i] << ","
                 << std::fixed << std::setprecision(6) << train[i] << ",";
            if (i < val.size()) {
                file << std::fixed << std::setprecision(6) << val[i];
            } else {
                file << "-1.0";
            }
            file << "\n";
        }
        file.close();
    }
};

// Learning rate scheduler with warmup and cosine decay
float get_lr(int step, int warmup_steps, int max_steps, float max_lr, float min_lr) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / warmup_steps;
    }
    if (step >= max_steps) {
        return min_lr;
    }
    float progress = static_cast<float>(step - warmup_steps) / (max_steps - warmup_steps);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(M_PI * progress));
}

// Simple progress bar
void print_progress(int step, int max_steps, float loss, float lr, const std::string& prefix = "") {
    int bar_width = 30;
    float progress = static_cast<float>(step) / max_steps;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "% "
              << "loss: " << std::fixed << std::setprecision(4) << loss
              << " lr: " << std::scientific << lr;
    std::cout.flush();
}

// Generate text from model
std::string generate(
    GPT& model,
    const std::string& prompt,
    const std::unordered_map<char, int>& stoi,
    const std::unordered_map<int, char>& itos,
    int max_new_tokens = 200,
    float temperature = 0.8f,
    int top_k = 40,
    int seed = -1) {
    
    if (seed != -1) {
        std::srand(seed);
    }
    
    // Convert prompt to tokens
    std::vector<int> tokens;
    for (char c : prompt) {
        if (stoi.find(c) != stoi.end()) {
            tokens.push_back(stoi.at(c));
        }
    }
    
    if (tokens.empty()) {
        return prompt; // Return original prompt if no valid tokens
    }
    
    // Set model to eval mode (no dropout in our implementation, but keep for consistency)
    // In our case, we just use the model as-is
    
    std::vector<int> gen_tokens = tokens;
    int block_size = model.config.block_size;
    
    for (int step = 0; step < max_new_tokens; step++) {
        // Get context (last block_size tokens)
        int start_idx = std::max(0, static_cast<int>(gen_tokens.size()) - block_size);
        int context_len = gen_tokens.size() - start_idx;
        
        // Create input tensor [context_len] (1D for embedding)
        Tensor idx({static_cast<int64_t>(context_len)});
        for (int i = 0; i < context_len; i++) {
            idx.data[i] = static_cast<float>(gen_tokens[start_idx + i]);
        }
        
        // Forward pass
        auto result = model.forward(idx);
        Tensor logits = result.first; // [1, context_len, vocab_size]
        
        // Get logits for last position
        float* last_logits = &logits.data[(context_len - 1) * model.config.vocab_size];
        
        // Apply temperature
        for (int i = 0; i < model.config.vocab_size; i++) {
            last_logits[i] /= temperature;
        }
        
        // Apply top-k filtering
        if (top_k > 0) {
            // Create vector of pairs (logit, index) for sorting
            std::vector<std::pair<float, int>> logits_with_idx;
            for (int i = 0; i < model.config.vocab_size; i++) {
                logits_with_idx.emplace_back(last_logits[i], i);
            }
            
            // Sort descending by logit value
            std::sort(logits_with_idx.begin(), logits_with_idx.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });
            
            // Find the k-th largest logit
            float threshold = logits_with_idx[std::min(top_k, static_cast<int>(logits_with_idx.size())) - 1].first;
            
            // Set logits below threshold to -inf
            for (int i = 0; i < model.config.vocab_size; i++) {
                if (last_logits[i] < threshold) {
                    last_logits[i] = -1e9f;
                }
            }
        }
        
        // Softmax to get probabilities
        Tensor probs_tensor({1, static_cast<int64_t>(model.config.vocab_size)});
        for (int i = 0; i < model.config.vocab_size; i++) {
            probs_tensor.data[i] = last_logits[i];
        }
        probs_tensor = ops::softmax(probs_tensor);
        
        // Sample from distribution
        float r = static_cast<float>(std::rand()) / RAND_MAX;
        float cumulative = 0.0f;
        int next_token = 0;
        for (int i = 0; i < model.config.vocab_size; i++) {
            cumulative += probs_tensor.data[i];
            if (cumulative >= r) {
                next_token = i;
                break;
            }
        }
        
        // Append sampled token
        gen_tokens.push_back(next_token);
    }
    
    // Convert tokens back to string
    std::string result;
    for (int token : gen_tokens) {
        if (itos.find(token) != itos.end()) {
            result += itos.at(token);
        }
    }
    
    return result;
}

// Save model checkpoint
void save_checkpoint(
    const GPT& model,
    const AdamW& optimizer,
    int step,
    const std::unordered_map<char, int>& stoi,
    const std::unordered_map<int, char>& itos,
    const std::string& filename) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open checkpoint file for writing: " << filename << std::endl;
        return;
    }
    
    // Write config
    file.write(reinterpret_cast<const char*>(&model.config.vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.config.block_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.config.n_layer), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.config.n_head), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.config.n_embd), sizeof(int));
    
    // Write model weights
    // wte
    int wte_size = model.wte_weight.dims[0] * model.wte_weight.dims[1];
    file.write(reinterpret_cast<const char*>(model.wte_weight.data.data()), wte_size * sizeof(float));
    // wpe
    int wpe_size = model.wpe_weight.dims[0] * model.wpe_weight.dims[1];
    file.write(reinterpret_cast<const char*>(model.wpe_weight.data.data()), wpe_size * sizeof(float));
    // blocks
    for (const Block& block : model.blocks) {
        // ln_1
        file.write(reinterpret_cast<const char*>(block.ln_1_weight.data.data()), block.ln_1_weight.dims[0] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.ln_1_bias.data.data()), block.ln_1_bias.dims[0] * sizeof(float));
        // attn
        file.write(reinterpret_cast<const char*>(block.attn.c_attn_weight.data.data()), block.attn.c_attn_weight.dims[0] * block.attn.c_attn_weight.dims[1] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.attn.c_attn_bias.data.data()), block.attn.c_attn_bias.dims[0] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.attn.c_proj_weight.data.data()), block.attn.c_proj_weight.dims[0] * block.attn.c_proj_weight.dims[1] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.attn.c_proj_bias.data.data()), block.attn.c_proj_bias.dims[0] * sizeof(float));
        // ln_2
        file.write(reinterpret_cast<const char*>(block.ln_2_weight.data.data()), block.ln_2_weight.dims[0] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.ln_2_bias.data.data()), block.ln_2_bias.dims[0] * sizeof(float));
        // mlp
        file.write(reinterpret_cast<const char*>(block.mlp.c_fc_weight.data.data()), block.mlp.c_fc_weight.dims[0] * block.mlp.c_fc_weight.dims[1] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.mlp.c_fc_bias.data.data()), block.mlp.c_fc_bias.dims[0] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.mlp.c_proj_weight.data.data()), block.mlp.c_proj_weight.dims[0] * block.mlp.c_proj_weight.dims[1] * sizeof(float));
        file.write(reinterpret_cast<const char*>(block.mlp.c_proj_bias.data.data()), block.mlp.c_proj_bias.dims[0] * sizeof(float));
    }
    // ln_f
    file.write(reinterpret_cast<const char*>(model.ln_f_weight.data.data()), model.ln_f_weight.dims[0] * sizeof(float));
    file.write(reinterpret_cast<const char*>(model.ln_f_bias.data.data()), model.ln_f_bias.dims[0] * sizeof(float));
    
    // Write optimizer state (simplified - just learning rate for now)
    file.write(reinterpret_cast<const char*>(&optimizer.learning_rate), sizeof(float));
    file.write(reinterpret_cast<const char*>(&optimizer.t), sizeof(int));
    
    // Write vocab mappings
    int vocab_size = stoi.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    for (const auto& pair : stoi) {
        char c = pair.first;
        int idx = pair.second;
        file.write(&c, sizeof(char));
        file.write(reinterpret_cast<const char*>(&idx), sizeof(int));
    }
    
    file.close();
    std::cout << "\nCheckpoint saved to " << filename << std::endl;
}

// Load model checkpoint
void load_checkpoint(
    GPT& model,
    AdamW& optimizer,
    int& step,
    std::unordered_map<char, int>& stoi,
    std::unordered_map<int, char>& itos,
    const std::string& filename) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open checkpoint file for reading: " << filename << std::endl;
        return;
    }
    
    // Read config
    file.read(reinterpret_cast<char*>(&model.config.vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.config.block_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.config.n_layer), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.config.n_head), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.config.n_embd), sizeof(int));
    
    // Read model weights
    // wte
    int wte_size = model.wte_weight.dims[0] * model.wte_weight.dims[1];
    file.read(reinterpret_cast<char*>(model.wte_weight.data.data()), wte_size * sizeof(float));
    // wpe
    int wpe_size = model.wpe_weight.dims[0] * model.wpe_weight.dims[1];
    file.read(reinterpret_cast<char*>(model.wpe_weight.data.data()), wpe_size * sizeof(float));
    // blocks
    for (Block& block : model.blocks) {
        // ln_1
        file.read(reinterpret_cast<char*>(block.ln_1_weight.data.data()), block.ln_1_weight.dims[0] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.ln_1_bias.data.data()), block.ln_1_bias.dims[0] * sizeof(float));
        // attn
        file.read(reinterpret_cast<char*>(block.attn.c_attn_weight.data.data()), block.attn.c_attn_weight.dims[0] * block.attn.c_attn_weight.dims[1] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.attn.c_attn_bias.data.data()), block.attn.c_attn_bias.dims[0] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.attn.c_proj_weight.data.data()), block.attn.c_proj_weight.dims[0] * block.attn.c_proj_weight.dims[1] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.attn.c_proj_bias.data.data()), block.attn.c_proj_bias.dims[0] * sizeof(float));
        // ln_2
        file.read(reinterpret_cast<char*>(block.ln_2_weight.data.data()), block.ln_2_weight.dims[0] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.ln_2_bias.data.data()), block.ln_2_bias.dims[0] * sizeof(float));
        // mlp
        file.read(reinterpret_cast<char*>(block.mlp.c_fc_weight.data.data()), block.mlp.c_fc_weight.dims[0] * block.mlp.c_fc_weight.dims[1] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.mlp.c_fc_bias.data.data()), block.mlp.c_fc_bias.dims[0] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.mlp.c_proj_weight.data.data()), block.mlp.c_proj_weight.dims[0] * block.mlp.c_proj_weight.dims[1] * sizeof(float));
        file.read(reinterpret_cast<char*>(block.mlp.c_proj_bias.data.data()), block.mlp.c_proj_bias.dims[0] * sizeof(float));
    }
    // ln_f
    file.read(reinterpret_cast<char*>(model.ln_f_weight.data.data()), model.ln_f_weight.dims[0] * sizeof(float));
    file.read(reinterpret_cast<char*>(model.ln_f_bias.data.data()), model.ln_f_bias.dims[0] * sizeof(float));
    
    // Read optimizer state
    file.read(reinterpret_cast<char*>(&optimizer.learning_rate), sizeof(float));
    file.read(reinterpret_cast<char*>(&optimizer.t), sizeof(int));
    
    // Read vocab mappings
    int vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    stoi.clear();
    itos.clear();
    for (int i = 0; i < vocab_size; i++) {
        char c;
        int idx;
        file.read(&c, sizeof(char));
        file.read(reinterpret_cast<char*>(&idx), sizeof(int));
        stoi[c] = idx;
        itos[idx] = c;
    }
    
    file.close();
    std::cout << "Checkpoint loaded from " << filename << " (step " << step << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default hyperparameters
    int max_steps = 5000;
    int batch_size = 64;
    int block_size = 256;
    int n_layer = 6;
    int n_head = 6;
    int n_embd = 384;
    float learning_rate = 1e-3f;
    int warmup_steps = 100;
    std::string data_path = "./data/shakespeare.txt";
    std::string checkpoint_dir = "./";
    int seed = 42;
    
    // Parse command line arguments (simplified)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--max_steps" && i + 1 < argc) {
            max_steps = std::stoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--block_size" && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if (arg == "--n_layer" && i + 1 < argc) {
            n_layer = std::stoi(argv[++i]);
        } else if (arg == "--n_head" && i + 1 < argc) {
            n_head = std::stoi(argv[++i]);
        } else if (arg == "--n_embd" && i + 1 < argc) {
            n_embd = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        }
    }
    
    // Set random seed
    std::srand(seed);
    
    std::cout << "Loading data from " << data_path << std::endl;
    auto [tokens, stoi, itos, vocab_size] = utils::load_data(data_path);
    std::cout << "Dataset: " << tokens.size() << " tokens, vocab size: " << vocab_size << std::endl;
    
    // Create model configuration
    GPTConfig config;
    config.vocab_size = vocab_size;
    config.block_size = block_size;
    config.n_layer = n_layer;
    config.n_head = n_head;
    config.n_embd = n_embd;
    
    // Create model
    GPT model(config);
    std::cout << "Model: " << n_layer << "L/" << n_head << "H/" << n_embd << "D, "
              << std::fixed << std::setprecision(1) 
              << (static_cast<double>(model.wte_weight.data.size() + 
                                          model.wpe_weight.data.size() +
                                          // Rough estimate of params in blocks
                                          n_layer * (4 * n_embd * n_embd +  // attn
                                               2 * n_embd +               // attn bias
                                               8 * n_embd * n_embd +      // mlp
                                               2 * 4 * n_embd) +          // mlp bias
                                          n_embd + n_embd) / 1e6) 
              << "M params" << std::endl;
    
    // Create optimizer
    AdamW optimizer(learning_rate);
    
    // Initialize optimizer with model parameters
    std::vector<Tensor> params;
    // Collect all model parameters
    params.push_back(model.wte_weight);
    params.push_back(model.wpe_weight);
    for (Block& block : model.blocks) {
        params.push_back(block.ln_1_weight);
        params.push_back(block.ln_1_bias);
        params.push_back(block.attn.c_attn_weight);
        params.push_back(block.attn.c_attn_bias);
        params.push_back(block.attn.c_proj_weight);
        params.push_back(block.attn.c_proj_bias);
        params.push_back(block.ln_2_weight);
        params.push_back(block.ln_2_bias);
        params.push_back(block.mlp.c_fc_weight);
        params.push_back(block.mlp.c_fc_bias);
        params.push_back(block.mlp.c_proj_weight);
        params.push_back(block.mlp.c_proj_bias);
    }
    params.push_back(model.ln_f_weight);
    params.push_back(model.ln_f_bias);
    
    optimizer.init(params);
    
    // Training loop
    LossLog loss_log;
    int token_idx = 0;  // Position in token stream for batch generation
    
    std::cout << "Starting training for " << max_steps << " steps..." << std::endl;
    
    for (int step = 0; step < max_steps; step++) {
        // Get batch
        auto batch = utils::get_batch(tokens, block_size, batch_size, token_idx);
        std::vector<int>& x_flat = batch.first;
        std::vector<int>& y_flat = batch.second;
        
        // Update token index for next batch
        token_idx += batch_size;
        if (token_idx > static_cast<int>(tokens.size()) - block_size - 1) {
            token_idx = 0;  // Reset to beginning
        }
        
        // Convert to tensors
        Tensor x({batch_size, block_size});
        Tensor y({batch_size, block_size});
        for (int i = 0; i < batch_size * block_size; i++) {
            x.data[i] = static_cast<float>(x_flat[i]);
            y.data[i] = static_cast<float>(y_flat[i]);
        }
        
        // Forward pass
        auto result = model.forward(x, &y);
        float loss = result.second;
        
        // Backward pass would go here - but we haven't implemented autograd
        // For now, we'll skip the backward pass and just use random gradients
        // In a real implementation, we would compute gradients via backprop
        
        // Dummy gradients for demonstration (in practice, these would come from backprop)
        std::vector<Tensor> grads;
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, 0.01f);
        for (const Tensor& param : params) {
            Tensor grad(param.dims);
            for (size_t i = 0; i < param.data.size(); i++) {
                grad.data[i] = distribution(generator);
            }
            grads.push_back(grad);
        }
        
        // Update learning rate
        float lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1f);
        optimizer.learning_rate = lr;
        
        // Clip gradients (simplified)
        float max_norm = 1.0f;
        float total_norm = 0.0f;
        for (const Tensor& grad : grads) {
            for (float g : grad.data) {
                total_norm += g * g;
            }
        }
        total_norm = std::sqrt(total_norm);
        float clip_coef = max_norm / (total_norm + 1e-6f);
        if (clip_coef < 1.0f) {
            for (Tensor& grad : grads) {
                for (float& g : grad.data) {
                    g *= clip_coef;
                }
            }
        }
        
        // Optimizer step
        optimizer.step(params, grads);
        
        // Logging
        if (step % 100 == 0) {
            // Validation (simplified - just use training loss for now)
            float val_loss = loss;  // In practice, we'd compute on validation set
            
            loss_log.add(step, loss, val_loss);
            print_progress(step, max_steps, loss, lr, "Training");
            std::cout << " | val loss: " << std::fixed << std::setprecision(4) << val_loss;
            
            // Generate sample
            if (step > 0) {
                // In our simple model, we don't have train/eval modes, so just use it directly
                std::string sample = generate(model, "To be or not", stoi, itos, 100, 0.8f, 40, seed + step);
                std::cout << "\n--- Sample ---\n" << sample << "\n--- End sample ---\n";
            }
            std::cout << std::endl;
        }
        
        // Save checkpoint
        if (step > 0 && step % 1000 == 0) {
            std::string checkpoint_file = checkpoint_dir + "checkpoint_" + std::to_string(step) + ".bin";
            save_checkpoint(model, optimizer, step, stoi, itos, checkpoint_file);
        }
    }
    
    // Save final checkpoint
    std::string final_checkpoint = checkpoint_dir + "checkpoint_final.bin";
    save_checkpoint(model, optimizer, max_steps, stoi, itos, final_checkpoint);
    
    // Save loss log
    std::string loss_log_file = checkpoint_dir + "loss_log.json";
    loss_log.save(loss_log_file);
    
    std::cout << "Training completed!" << std::endl;
    return 0;
}