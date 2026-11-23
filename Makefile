CXX ?= c++
CXXFLAGS ?= -O2
BUILD_DIR ?= build

HOST_OS := $(shell uname -s)
METAL ?= $(if $(filter Darwin,$(HOST_OS)),1,0)
ACCELERATE ?= $(if $(filter Darwin,$(HOST_OS)),1,0)
OPENMP ?= 0

ifeq ($(METAL),1)
ifneq ($(HOST_OS),Darwin)
$(error METAL=1 requires macOS)
endif
BACKEND_FLAGS += -DTINYNANOGPT_USE_METAL -fobjc-arc
BACKEND_LIBS += -framework Foundation -framework Metal -framework MetalPerformanceShaders
BACKEND_SOURCES += metal_backend.mm
endif

ifeq ($(ACCELERATE),1)
ifneq ($(HOST_OS),Darwin)
$(error ACCELERATE=1 requires macOS)
endif
BACKEND_FLAGS += -DTINYNANOGPT_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
BACKEND_LIBS += -framework Accelerate
endif

ifeq ($(OPENMP),1)
ifeq ($(HOST_OS),Darwin)
OPENMP_FLAGS ?= -Xpreprocessor -fopenmp
OPENMP_LIBS ?= -lomp
else
OPENMP_FLAGS ?= -fopenmp
OPENMP_LIBS ?= -fopenmp
endif
BACKEND_FLAGS += $(OPENMP_FLAGS)
BACKEND_LIBS += $(OPENMP_LIBS)
endif

.PHONY: all
all: $(BUILD_DIR)/train $(BUILD_DIR)/generate

$(BUILD_DIR)/%: %.cpp model.h matrix_backend.h $(BACKEND_SOURCES) Makefile | $(BUILD_DIR)
	$(CXX) -std=c++17 $(CPPFLAGS) $(CXXFLAGS) $(BACKEND_FLAGS) $< $(BACKEND_SOURCES) $(LDFLAGS) $(BACKEND_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

ifeq ($(HOST_OS),Darwin)
.PHONY: metal-benchmark
metal-benchmark: $(BUILD_DIR)/metal-train-bench

$(BUILD_DIR)/metal-train-bench: benchmarks/metal_train_bench.mm model.h matrix_backend.h Makefile | $(BUILD_DIR)
	$(CXX) -std=c++17 $(CPPFLAGS) $(CXXFLAGS) -fobjc-arc -I. $< $(LDFLAGS) \
	  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
	  -framework MetalPerformanceShadersGraph $(LDLIBS) -o $@
endif
