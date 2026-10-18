#include "model.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <algorithm>

// Function to load checkpoint
void load_checkpoint(GPT& model, AdamW& optimizer, int& step, std::unordered_map<char, int>& stoi, std::unordered_map<int, char>& itos, const std::string& filename) {
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

// Function to generate text
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

        // Create input tensor [1, context_len]
        Tensor idx({1, static_cast<int64_t>(context_len)});
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

int main(int argc, char* argv[]) {
    // Default parameters
    std::string checkpoint_path = "./checkpoint_final.bin";
    std::string prompt = "To be or not";
    int max_new_tokens = 200;
    float temperature = 0.8f;
    int top_k = 40;
    int seed = -1;  // -1 means random seed

    // Parse command line arguments (simplified)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max_new_tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        }
    }

    // Set seed if provided
    if (seed != -1) {
        std::srand(seed);
    } else {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    std::cout << "Loading checkpoint from " << checkpoint_path << std::endl;

    // We need to load the checkpoint to get the model and vocab mappings
    // Since our load_checkpoint function requires a model to be constructed first,
    // we'll create a dummy model with default config and then load the checkpoint

    // First, let's try to read the config from the checkpoint file to create the proper model
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open checkpoint file: " << checkpoint_path << std::endl;
        return 1;
    }

    // Read config from checkpoint
    GPTConfig config;
    file.read(reinterpret_cast<char*>(&config.vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&config.block_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&config.n_layer), sizeof(int));
    file.read(reinterpret_cast<char*>(&config.n_head), sizeof(int));
    file.read(reinterpret_cast<char*>(&config.n_embd), sizeof(int));
    file.close();

    // Create model with loaded config
    GPT model(config);

    // Create dummy optimizer and vocab mappings for loading
    AdamW optimizer;
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    int dummy_step = 0;

    // Load the checkpoint
    load_checkpoint(model, optimizer, dummy_step, stoi, itos, checkpoint_path);

    std::cout << "Generating text..." << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Temperature: " << temperature << ", Top-k: " << top_k << std::endl;
    if (seed != -1) {
        std::cout << "Seed: " << seed << std::endl;
    }

    // Generate text
    std::string output = generate(model, prompt, stoi, itos, max_new_tokens, temperature, top_k, seed);

    std::cout << "\nGenerated text:\n";
    std::cout << output << std::endl;

    return 0;
}