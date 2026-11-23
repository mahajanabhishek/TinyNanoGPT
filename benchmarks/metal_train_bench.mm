// Standalone, GPU-resident benchmark for the first steps of Shakespeare training.
#include "model.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <chrono>
#include <stdexcept>
#include <string>

constexpr int time_steps = 256, channels = 384, heads = 6, layers = 6;

using GraphTensor = MPSGraphTensor*;

template <typename T>
NSData* bytes_of(const std::vector<T>& data) {
    return [NSData dataWithBytes:data.data() length:data.size() * sizeof(T)];
}

float read_loss(GraphTensor loss, MPSGraphTensorDataDictionary* results, id<MTLCommandQueue> queue) {
    MPSNDArray* array = [results[loss] mpsndarray];
    if (!array) throw std::runtime_error("Loss tensor was not returned");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    [array synchronizeOnCommandBuffer:command];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("Loss readback failed");
    float loss_value = 0;
    [array readBytes:&loss_value strideBytes:nullptr];
    return loss_value;
}

void seed_weights(GPT& model) {
    std::vector<Tensor*> weights{&model.wte_weight, &model.wpe_weight};
    for (Block& block : model.blocks) {
        weights.insert(weights.end(), {&block.attn.c_attn_weight, &block.attn.c_proj_weight,
                                       &block.mlp.c_fc_weight, &block.mlp.c_proj_weight});
    }
    std::mt19937 random(42);
    std::normal_distribution<double> normal(0.0, 0.02);
    for (Tensor* weight : weights) {
        for (float& value : weight->data) value = static_cast<float>(normal(random));
    }
}

void report(int step, double seconds, float loss) {
    if (!std::isfinite(loss)) throw std::runtime_error("Nonfinite loss");
    std::cout << "STEP " << step << ' ' << seconds << ' ' << loss << std::endl;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 4) throw std::invalid_argument("Usage: metal-train-bench DATA BATCH STEPS");
            const int batch = std::stoi(argv[2]), steps = std::stoi(argv[3]);
            if (batch < 1 || batch > 64 || steps < 1 || steps > 100) {
                throw std::invalid_argument("Expected batch size 1–64 and 1–100 warmup steps");
            }
            const auto [tokens, stoi, itos, vocabulary] = utils::load_data(argv[1]);
            if (tokens.size() < static_cast<size_t>(steps * batch + time_steps)) {
                throw std::invalid_argument("Training text is too short");
            }
            GPT model({vocabulary, time_steps, layers, heads, channels});
            seed_weights(model);
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (!metal || !MPSSupportsMTLDevice(metal)) throw std::runtime_error("Metal GPU not available");
            MPSGraphDevice* device = [MPSGraphDevice deviceWithMTLDevice:metal];
            id<MTLCommandQueue> queue = [metal newCommandQueue];
            if (!queue) throw std::runtime_error("Could not create a Metal queue");
            std::cout << "Device: " << metal.name.UTF8String << std::endl;

            MPSGraph* graph = [MPSGraph new];
            graph.options = MPSGraphOptionsNone;
            auto feeds = [NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*> dictionary];
            auto parameters = [NSMutableArray<MPSGraphTensor*> array];
            auto placeholder = [&](NSArray<NSNumber*>* shape, MPSDataType type, NSData* initial) {
                GraphTensor tensor = [graph placeholderWithShape:shape dataType:type name:nil];
                MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithDevice:device
                                            data:initial shape:shape dataType:type];
                if (!data) throw std::runtime_error("Could not allocate GPU tensor data");
                feeds[tensor] = data;
                return tensor;
            };
            auto param = [&](const Tensor& values) {
                auto shape = [NSMutableArray<NSNumber*> array];
                for (auto dim : values.dims) [shape addObject:@(dim)];
                const auto tensor = placeholder(shape, MPSDataTypeFloat32, bytes_of(values.data));
                [parameters addObject:tensor];
                return tensor;
            };
            auto scalar = [&](float value) {
                return [graph constantWithScalar:value dataType:MPSDataTypeFloat32];
            };
            auto plus = [&](GraphTensor a, GraphTensor b) {
                return [graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
            };
            auto minus = [&](GraphTensor a, GraphTensor b) {
                return [graph subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
            };
            auto times = [&](GraphTensor a, GraphTensor b) {
                return [graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
            };
            auto divide = [&](GraphTensor a, GraphTensor b) {
                return [graph divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
            };
            auto sqrt = [&](GraphTensor x) { return [graph squareRootWithTensor:x name:nil]; };
            auto matmul = [&](GraphTensor a, GraphTensor b) {
                auto a16 = [graph castTensor:a toType:MPSDataTypeFloat16 name:nil];
                auto b16 = [graph castTensor:b toType:MPSDataTypeFloat16 name:nil];
                auto product = [graph matrixMultiplicationWithPrimaryTensor:a16 secondaryTensor:b16 name:nil];
                return [graph castTensor:product toType:MPSDataTypeFloat32 name:nil];
            };
            auto swap = [&](GraphTensor x, int first, int second) {
                return [graph transposeTensor:x dimension:first withDimension:second name:nil];
            };
            auto reshape = [&](GraphTensor x, NSArray<NSNumber*>* shape) {
                return [graph reshapeTensor:x withShape:shape name:nil];
            };
            auto reduce = [&](GraphTensor x, NSArray<NSNumber*>* axes) {
                return [graph reductionSumWithTensor:x axes:axes name:nil];
            };
            auto norm = [&](GraphTensor x, const Tensor& weight, const Tensor& bias) {
                auto centered = minus(x, times(reduce(x, @[@2]), scalar(1.0f / channels)));
                auto variance = times(reduce(times(centered, centered), @[@2]), scalar(1.0f / channels));
                return plus(times(divide(centered, sqrt(plus(variance, scalar(1e-5f)))), param(weight)), param(bias));
            };
            auto dense_product = [&](GraphTensor x, GraphTensor weight, int64_t inputs, int64_t outputs) {
                auto flat = reshape(x, @[@(batch * time_steps), @(inputs)]);
                auto product = matmul(flat, weight);
                return reshape(product, @[@(batch), @(time_steps), @(outputs)]);
            };
            auto dense = [&](GraphTensor x, const Tensor& weight, const Tensor& bias) {
                return plus(dense_product(x, param(weight), weight.dims[0], weight.dims[1]), param(bias));
            };

            auto input_shape = @[@(batch), @(time_steps)];
            auto make_input = [&](const std::vector<int>& ids) {
                return [[MPSGraphTensorData alloc] initWithDevice:device
                    data:bytes_of(ids) shape:input_shape dataType:MPSDataTypeInt32];
            };
            auto token_ids = [graph placeholderWithShape:input_shape dataType:MPSDataTypeInt32 name:@"tokens"];
            auto target_ids = [graph placeholderWithShape:input_shape dataType:MPSDataTypeInt32 name:@"targets"];
            auto onehot = [graph oneHotWithIndicesTensor:target_ids depth:vocabulary axis:2
                                                dataType:MPSDataTypeFloat32 name:nil];
            auto wte = param(model.wte_weight);
            auto positions = param(model.wpe_weight);
            auto hidden = plus([graph gatherWithUpdatesTensor:wte indicesTensor:token_ids axis:0 batchDimensions:0 name:nil],
                               positions);

            std::vector<float> mask(time_steps * time_steps);
            for (int row = 0; row < time_steps; ++row) {
                for (int col = row + 1; col < time_steps; ++col) mask[row * time_steps + col] = -1e9f;
            }
            auto causal_mask = [graph constantWithData:bytes_of(mask)
                                                 shape:@[@(time_steps), @(time_steps)] dataType:MPSDataTypeFloat32];

            for (const Block& block : model.blocks) {
                auto qkv = dense(norm(hidden, block.ln_1_weight, block.ln_1_bias),
                                 block.attn.c_attn_weight, block.attn.c_attn_bias);
                auto head_tensor = [&](int component) {
                    auto part = [graph sliceTensor:qkv dimension:2 start:component * channels length:channels name:nil];
                    return swap(reshape(part, @[@(batch), @(time_steps), @(heads), @(channels / heads)]), 1, 2);
                };
                auto q = head_tensor(0), k = head_tensor(1), v = head_tensor(2);
                const float scale = 1.0f / std::sqrt(channels / heads);
                auto scores = times(matmul(q, swap(k, 2, 3)), scalar(scale));
                auto probabilities = [graph softMaxWithTensor:plus(scores, causal_mask) axis:3 name:nil];
                auto attention = matmul(probabilities, v);
                auto context = reshape(swap(attention, 1, 2), @[@(batch), @(time_steps), @(channels)]);
                hidden = plus(hidden, dense(context, block.attn.c_proj_weight, block.attn.c_proj_bias));

                auto fc = dense(norm(hidden, block.ln_2_weight, block.ln_2_bias),
                                block.mlp.c_fc_weight, block.mlp.c_fc_bias);
                auto gelu = times(times(scalar(0.5f), fc),
                                  plus(scalar(1.0f), [graph erfWithTensor:times(fc, scalar(0.7071067812f)) name:nil]));
                hidden = plus(hidden, dense(gelu, block.mlp.c_proj_weight, block.mlp.c_proj_bias));
            }
            auto logits = dense_product(norm(hidden, model.ln_f_weight, model.ln_f_bias),
                                        swap(wte, 0, 1), channels, vocabulary);
            auto loss = [graph softMaxCrossEntropyWithSourceTensor:logits labelsTensor:onehot axis:2
                                                    reductionType:MPSGraphLossReductionTypeMean name:nil];

            constexpr float loss_scale = 1024.0f;
            auto differentiation_loss = times(loss, scalar(loss_scale));
            auto raw_gradients = [graph gradientForPrimaryTensor:differentiation_loss withTensors:parameters name:nil];
            auto gradients = [NSMutableDictionary<MPSGraphTensor*, MPSGraphTensor*> dictionary];
            for (GraphTensor weight in parameters) {
                auto raw = raw_gradients[weight];
                if (!raw) throw std::runtime_error("Missing parameter gradient");
                gradients[weight] = times(raw, scalar(1.0f / loss_scale));
            }
            auto updates = [NSMutableDictionary<MPSGraphTensor*, MPSGraphTensor*> dictionary];
            auto results_to_keep = [NSMutableArray<MPSGraphTensor*> arrayWithObject:loss];
            auto record = [&](GraphTensor previous, GraphTensor updated) {
                updates[previous] = updated;
                [results_to_keep addObject:updated];
            };
            auto step = placeholder(@[@1], MPSDataTypeFloat32, bytes_of(std::vector<float>{0.0f}));
            auto next_step = plus(step, scalar(1.0f));
            record(step, next_step);
            auto learning_rate = times(scalar(1e-5f), next_step);
            auto power = [&](float base) {
                return [graph powerWithPrimaryTensor:scalar(base) secondaryTensor:next_step name:nil];
            };
            auto norm2 = scalar(0.0f);
            for (GraphTensor weight in parameters) {
                auto grad = gradients[weight];
                if (!grad) throw std::runtime_error("Missing parameter gradient");
                auto axes = weight.shape.count == 1 ? @[@0] : @[@0, @1];
                norm2 = plus(norm2, reshape(reduce(times(grad, grad), axes), @[@1]));
            }
            auto clip = [graph minimumWithPrimaryTensor:scalar(1.0f)
                secondaryTensor:divide(scalar(1.0f), plus(sqrt(norm2), scalar(1e-6f))) name:nil];
            auto correction1 = minus(scalar(1.0f), power(0.9f));
            auto correction2 = minus(scalar(1.0f), power(0.999f));
            for (GraphTensor weight in parameters) {
                size_t size = 1;
                for (NSNumber* dim in weight.shape) size *= dim.unsignedLongLongValue;
                NSData* zeros = [NSData dataWithBytes:std::vector<float>(size).data() length:size * sizeof(float)];
                auto first = placeholder(weight.shape, MPSDataTypeFloat32, zeros);
                auto second = placeholder(weight.shape, MPSDataTypeFloat32, zeros);
                auto grad = times(gradients[weight], clip);
                auto m = plus(times(scalar(0.9f), first), times(scalar(1.0f - 0.9f), grad));
                auto v2 = plus(times(scalar(0.999f), second), times(scalar(1.0f - 0.999f), times(grad, grad)));
                auto direction = divide(divide(m, correction1),
                                        plus(sqrt(divide(v2, correction2)), scalar(1e-8f)));
                record(first, m);
                record(second, v2);
                record(weight, minus(weight, times(learning_rate, plus(direction, times(scalar(0.01f), weight)))));
            }

            std::cout << "Built " << layers << "-layer graph with batch " << batch
                      << " and context " << time_steps << ". Compiling on first step." << std::endl;
            for (int i = 0; i < steps; ++i) {
                @autoreleasepool {
                    const auto start = std::chrono::steady_clock::now();
                    const auto [x, y] = utils::get_batch(tokens, time_steps, batch, i * batch);
                    feeds[token_ids] = make_input(x);
                    feeds[target_ids] = make_input(y);
                    MPSCommandBuffer* command = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
                    auto results = [graph encodeToCommandBuffer:command feeds:feeds targetTensors:results_to_keep
                                               targetOperations:nil executionDescriptor:nil];
                    [command commit];
                    [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted) {
                        throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "GPU command failed");
                    }
                    const float loss_value = read_loss(loss, results, queue);
                    for (GraphTensor previous in updates) {
                        // The next iteration consumes these Metal tensor objects directly.
                        if (!results[updates[previous]]) throw std::runtime_error("Updated GPU tensor was not returned");
                        feeds[previous] = results[updates[previous]];
                    }
                    report(i + 1, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                           loss_value);
                }
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
