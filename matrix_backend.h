#ifndef TINYNANOGPT_MATRIX_BACKEND_H
#define TINYNANOGPT_MATRIX_BACKEND_H

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>

namespace matrix_backend {

enum class Device { cpu, automatic, metal };

#if defined(TINYNANOGPT_USE_METAL) && defined(__APPLE__)
std::string metal_device_name();
bool metal_matmul(const float* left, const float* right, float* output,
                  std::size_t rows, std::size_t columns, std::size_t inner, bool force);
#endif

inline std::atomic<Device>& selected_device() {
    static std::atomic<Device> selected{Device::cpu};
    return selected;
}

inline bool configure(const std::string& option) {
    if (option != "auto" && option != "cpu" && option != "metal") {
        std::cerr << "Expected --device auto, cpu, or metal\n";
        return false;
    }

#if defined(TINYNANOGPT_USE_METAL) && defined(__APPLE__)
    if (option != "cpu") {
        const auto name = metal_device_name();
        if (!name.empty()) {
            selected_device().store(option == "metal" ? Device::metal : Device::automatic);
            std::cerr << "Matrix backend: Metal (" << name << ") with CPU fallback\n";
            return true;
        }
        std::cerr << "Metal device unavailable. Using CPU.\n";
    }
#else
    if (option == "metal") std::cerr << "Metal support was not built. Using CPU.\n";
#endif

    selected_device().store(Device::cpu);
#if defined(TINYNANOGPT_USE_ACCELERATE) && defined(__APPLE__)
    std::cerr << "Matrix backend: CPU (Accelerate)\n";
#else
    std::cerr << "Matrix backend: CPU\n";
#endif
    return true;
}

#if defined(TINYNANOGPT_USE_METAL) && defined(__APPLE__)
inline bool try_metal_matmul(const float* left, const float* right, float* output,
                            std::size_t rows, std::size_t columns, std::size_t inner) {
    const auto device = selected_device().load(std::memory_order_relaxed);
    return device != Device::cpu &&
        metal_matmul(left, right, output, rows, columns, inner, device == Device::metal);
}
#endif

} // namespace matrix_backend

#endif
