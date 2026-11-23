#include "matrix_backend.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <array>
#include <cstring>
#include <mutex>

namespace matrix_backend {
namespace {

id<MTLDevice> find_device() {
    id<MTLDevice> preferred = MTLCreateSystemDefaultDevice();
    if (preferred && MPSSupportsMTLDevice(preferred)) return preferred;

    for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
        if (MPSSupportsMTLDevice(candidate)) return candidate;
    }
    return nil;
}

class MetalBackend {
public:
    MetalBackend() {
        @autoreleasepool {
            device = find_device();
            queue = [device newCommandQueue];
            if (device) managed = !device.hasUnifiedMemory;
        }
    }

    std::string name() {
        std::lock_guard<std::mutex> guard(mutex);
        return queue && !disabled && device.name.UTF8String ? device.name.UTF8String : "";
    }

    bool multiply(const float* left, const float* right, float* output,
                  std::size_t rows, std::size_t columns, std::size_t inner, bool force) {
        // Small matrix products rarely repay command submission or PCIe transfers.
        const double multiply_adds = static_cast<double>(rows) * columns * inner;
        if (!force && multiply_adds < (managed ? 1e9 : 5e7)) return false;

        std::lock_guard<std::mutex> guard(mutex);
        if (!queue || disabled) return false;

        @autoreleasepool {
            @try {
                MPSMatrix* a = matrix(0, rows, inner);
                MPSMatrix* b = matrix(1, inner, columns);
                MPSMatrix* c = matrix(2, rows, columns);
                if (!a || !b || !c) return false;

                upload(a, left);
                upload(b, right);
                MPSMatrixMultiplication* multiply = [[MPSMatrixMultiplication alloc] initWithDevice:device
                    transposeLeft:NO transposeRight:NO resultRows:rows resultColumns:columns
                    interiorColumns:inner alpha:1.0 beta:0.0];
                id<MTLCommandBuffer> command = [queue commandBuffer];
                if (!multiply || !command) return false;
                [multiply encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:c];

                if (managed) {
                    const auto bytes = rows * c.rowBytes;
                    if (!reserve(readback, bytes, MTLResourceStorageModeShared)) return false;
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    if (!blit) return false;
                    [blit copyFromBuffer:c.data sourceOffset:0 toBuffer:readback destinationOffset:0 size:bytes];
                    [blit endEncoding];
                }
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) {
                    const char* message = command.error.localizedDescription.UTF8String;
                    disable(message);
                    return false;
                }

                const auto source = managed ? readback.contents : c.data.contents;
                copy_rows(output, columns * sizeof(float), source, c.rowBytes, rows, columns * sizeof(float));
                return true;
            } @catch (NSException* error) {
                disable(error.reason.UTF8String);
                return false;
            }
        }
    }

private:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    // Reuse the two input buffers and the result buffer across products.
    std::array<id<MTLBuffer>, 3> buffers{};
    id<MTLBuffer> readback = nil;
    std::mutex mutex;
    bool managed = false;
    bool disabled = false;

    MPSMatrix* matrix(std::size_t slot, std::size_t height, std::size_t width) {
        if (height == 0 || width == 0 || width > device.maxBufferLength / sizeof(float)) return nil;
        const auto pitch = [MPSMatrixDescriptor rowBytesForColumns:width dataType:MPSDataTypeFloat32];
        if (pitch == 0 || height > device.maxBufferLength / pitch) return nil;
        MTLResourceOptions mode = MTLResourceStorageModeShared;
        if (managed) mode = slot == 2 ? MTLResourceStorageModePrivate : MTLResourceStorageModeManaged;
        if (!reserve(buffers[slot], height * pitch, mode)) return nil;
        MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:height
            columns:width rowBytes:pitch dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:buffers[slot] descriptor:descriptor];
    }

    bool reserve(id<MTLBuffer> __strong& buffer, std::size_t bytes, MTLResourceOptions mode) {
        if (buffer.length >= bytes) return true;
        id<MTLBuffer> replacement = [device newBufferWithLength:bytes options:mode];
        if (!replacement) return false;
        buffer = replacement;
        return true;
    }

    static void copy_rows(void* destination, std::size_t destination_stride,
                          const void* source, std::size_t source_stride,
                          std::size_t rows, std::size_t row_bytes) {
        auto* dst = static_cast<unsigned char*>(destination);
        const auto* src = static_cast<const unsigned char*>(source);
        if (destination_stride == row_bytes && source_stride == row_bytes) {
            std::memcpy(dst, src, rows * row_bytes);
            return;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            std::memcpy(dst + row * destination_stride, src + row * source_stride, row_bytes);
        }
    }

    void upload(MPSMatrix* matrix, const float* values) const {
        const auto bytes = matrix.columns * sizeof(float);
        copy_rows(matrix.data.contents, matrix.rowBytes, values, bytes, matrix.rows, bytes);
        if (managed) [matrix.data didModifyRange:NSMakeRange(0, matrix.rows * matrix.rowBytes)];
    }

    void disable(const char* reason) {
        disabled = true;
        std::cerr << "Metal matrix multiplication failed. Using CPU";
        if (reason) std::cerr << " (" << reason << ")";
        std::cerr << ".\n";
    }
};

MetalBackend& backend() {
    static MetalBackend instance;
    return instance;
}

} // namespace

std::string metal_device_name() {
    return backend().name();
}

bool metal_matmul(const float* left, const float* right, float* output,
                  std::size_t rows, std::size_t columns, std::size_t inner, bool force) {
    return backend().multiply(left, right, output, rows, columns, inner, force);
}

} // namespace matrix_backend
