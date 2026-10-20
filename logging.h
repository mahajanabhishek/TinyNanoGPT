#ifndef GPT_LOGGING_H
#define GPT_LOGGING_H

// ============================================================================
// Logging + progress reporting
// ----------------------------------------------------------------------------
// There are two independent knobs.
//
// 1. COMPILE-TIME LEVEL (costs nothing when off)
//
//        g++ -std=c++17 -O2 -DGPT_LOG_LEVEL=0 ...
//
//        0 = silent   1 = error   2 = warn   3 = info   4 = debug   5 = trace
//
//    Every log statement above the compile-time level is removed by the
//    preprocessor, so a build with -DGPT_LOG_LEVEL=0 contains no I/O calls and
//    no branch checks at all. This is what removes the hot-path diagnostics
//    (e.g. the per-call tensor dump in ops::add) from the training loop.
//
// 2. RUNTIME LEVEL (cheap, optional)
//
//        gpt::log::set_level(0);      // silence without recompiling
//        ./train --log-level 0        // ...or from the command line
//
//    The runtime level can only lower verbosity below the compile-time ceiling;
//    it can never re-enable statements that were compiled out.
// ============================================================================

#include <cstdio>
#include <chrono>
#include <string>
#include <cstdarg>
#include <cstdint>

#if defined(_WIN32)
#  include <io.h>
#else
#  include <unistd.h>
#endif

// Forward declare Tensor (defined in model.h)
class Tensor;

// ---------------------------------------------------------------------------
// Compile-time configuration
// ---------------------------------------------------------------------------
#ifndef GPT_LOG_LEVEL
#define GPT_LOG_LEVEL 3          // default: errors, warnings, info, progress
#endif

// Set to 0 to force the progress bar off at compile time as well.
#ifndef GPT_ENABLE_PROGRESS
#define GPT_ENABLE_PROGRESS 1
#endif

namespace gpt {
namespace log {

enum Level {
    SILENT = 0,
    ERROR  = 1,
    WARN   = 2,
    INFO   = 3,
    DEBUG  = 4,
    TRACE  = 5
};

// Runtime verbosity ceiling, initialised from the compile-time level.
inline int& level() {
    static int lvl = GPT_LOG_LEVEL;
    return lvl;
}

inline void set_level(int lvl) {
    level() = lvl < 0 ? 0 : (lvl > TRACE ? TRACE : lvl);
}

inline bool enabled(int lvl) {
    return GPT_LOG_LEVEL > 0 && lvl <= level();
}

// Is stdout a terminal? When it is not (output piped to a file or through
// `tail`), the progress bar falls back to periodic full lines instead of
// hammering the stream with carriage returns.
inline bool stdout_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    static const bool tty = ::isatty(fileno(stdout)) != 0;
    return tty;
#endif
}

// Minimal printf-style writer; avoids iostream formatting cost in hot paths.
inline void emit(const char* tag, const char* fmt, va_list ap) {
    std::fputs(tag, stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
}

// Helper to format tensor shape for logging
namespace gpt {
namespace log {
    // Forward declarations; definitions are in model.h after Tensor is defined
    std::string fmt_shape(const ::gpt::Tensor& t);
    std::string fmt_tensor(const ::gpt::Tensor& t);
} // namespace log
} // namespace gpt

// Compile-time assert that only triggers when logging is enabled.
// If logging is off at compile time, this becomes a no-op; otherwise it
// forwards to the standard assert so we still get file/line info.
#ifdef NDEBUG
#  define GPT_LOG_ASSERT(cond, msg) do {} while (0)
#else
#  if GPT_LOG_LEVEL >= 1
#    define GPT_LOG_ASSERT(cond, msg) do { if (!(cond)) { gpt::log::error("%s: %s", __func__, msg); assert(cond); } } while (0)
#  else
#    define GPT_LOG_ASSERT(cond, msg) do { if (!(cond)) assert(cond); } while (0)
#  endif
#endif

inline void error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); emit("[error] ", fmt, ap); va_end(ap);
}

inline void warn(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); emit("[warn]  ", fmt, ap); va_end(ap);
}

inline void info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); emit("[info]  ", fmt, ap); va_end(ap);
}

inline void debug(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); emit("[debug] ", fmt, ap); va_end(ap);
}

inline void trace(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); emit("[trace] ", fmt, ap); va_end(ap);
}

} // namespace log
} // namespace gpt

// ---------------------------------------------------------------------------
// Compile-time-gated macros
// ---------------------------------------------------------------------------
#if GPT_LOG_LEVEL >= 1
#define GPT_LOG_ERROR(...) do { if (gpt::log::enabled(gpt::log::ERROR)) gpt::log::error(__VA_ARGS__); } while (0)
#else
#define GPT_LOG_ERROR(...) do {} while (0)
#endif

#if GPT_LOG_LEVEL >= 2
#define GPT_LOG_WARN(...)  do { if (gpt::log::enabled(gpt::log::WARN))  gpt::log::warn(__VA_ARGS__);  } while (0)
#else
#define GPT_LOG_WARN(...)  do {} while (0)
#endif

#if GPT_LOG_LEVEL >= 3
#define GPT_LOG_INFO(...)  do { if (gpt::log::enabled(gpt::log::INFO))  gpt::log::info(__VA_ARGS__);  } while (0)
#else
#define GPT_LOG_INFO(...)  do {} while (0)
#endif

#if GPT_LOG_LEVEL >= 4
#define GPT_LOG_DEBUG(...) do { if (gpt::log::enabled(gpt::log::DEBUG)) gpt::log::debug(__VA_ARGS__); } while (0)
#else
#define GPT_LOG_DEBUG(...) do {} while (0)
#endif

#if GPT_LOG_LEVEL >= 5
#define GPT_LOG_TRACE(...) do { if (gpt::log::enabled(gpt::log::TRACE)) gpt::log::trace(__VA_ARGS__); } while (0)
#else
#define GPT_LOG_TRACE(...) do {} while (0)
#endif

// ===========================================================================
// Progress bar
// ---------------------------------------------------------------------------
// Renders `label [=====>    ]  42% 1234/5000  loss 3.21  lr 1.0e-03  12.3 tok/s  eta 00:02:31`
//
// * Updates are rate limited to `min_interval_ms` (default 100 ms), so the I/O
//   cost is bounded no matter how large the step count is.
// * On a TTY it rewrites a single line with '\r'.
// * When stdout is redirected it prints a full line at most once per second,
//   which keeps log files readable and cheap to tail.
// ===========================================================================
namespace gpt {

class ProgressBar {
public:
    ProgressBar(int64_t total, std::string label, bool enabled = true,
                int min_interval_ms = 100)
        : total_(total > 0 ? total : 1),
          label_(std::move(label)),
          enabled_(enabled && GPT_ENABLE_PROGRESS != 0),
          min_interval_(std::chrono::milliseconds(min_interval_ms)),
          start_(Clock::now()),
          last_(start_) {}

    ~ProgressBar() {
        if (enabled_ && !finished_) finish_line();
    }

    // Advance the bar. `suffix` is free-form status text appended to the line,
    // e.g. printf-formatted loss/lr/throughput.
    void update(int64_t current, const std::string& suffix = "") {
        if (!enabled_) return;

        const auto now = Clock::now();
        const auto since = now - last_;
        const bool done = current >= total_;

        // Always render the final state; otherwise respect the rate limit.
        if (!done && since < min_interval_) return;

        // When redirected, don't spam: one line per second unless finished.
        if (!done && !gpt::log::stdout_is_tty() && since < std::chrono::seconds(1)) return;

        last_ = now;
        render(current, suffix, done);

        if (done) { finished_ = true; finish_line(); }
    }

    void finish(const std::string& suffix = "") {
        if (!enabled_ || finished_) return;
        render(total_, suffix, true);
        finished_ = true;
        finish_line();
    }

    // Disable at construction time based on flags (e.g. --no-progress).
    void set_enabled(bool on) { enabled_ = on && GPT_ENABLE_PROGRESS != 0; }
    bool enabled() const { return enabled_; }

private:
    using Clock = std::chrono::steady_clock;

    static std::string fmt_duration(double seconds) {
        if (seconds < 0 || seconds > 359999.0) return "--:--:--";
        int64_t s = static_cast<int64_t>(seconds);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld",
                      static_cast<long long>(s / 3600),
                      static_cast<long long>((s % 3600) / 60),
                      static_cast<long long>(s % 60));
        return buf;
    }

    void render(int64_t current, const std::string& suffix, bool done) {
        if (current > total_) current = total_;

        const double frac = static_cast<double>(current) / static_cast<double>(total_);
        const int width = 24;
        const int filled = static_cast<int>(frac * width);

        const double elapsed =
            std::chrono::duration<double>(Clock::now() - start_).count();
        const double eta = (frac > 0.0001 && !done)
                               ? elapsed * (1.0 - frac) / frac
                               : 0.0;

        char line[512];
        int n = std::snprintf(
            line, sizeof(line),
            "%s [", label_.c_str());
        for (int i = 0; i < width && n < static_cast<int>(sizeof(line)) - 1; ++i) {
            line[n++] = (i < filled) ? '=' : (i == filled && !done ? '>' : ' ');
        }
        n += std::snprintf(line + n, sizeof(line) - n,
                           "] %3d%% %lld/%lld  %s  eta %s%s%s",
                           static_cast<int>(frac * 100.0),
                           static_cast<long long>(current),
                           static_cast<long long>(total_),
                           fmt_duration(elapsed).c_str(),
                           done ? "00:00:00" : fmt_duration(eta).c_str(),
                           suffix.empty() ? "" : "  ",
                           suffix.c_str());
        if (n < 0) return;
        line_ = line;
        rendered_ = true;
    }

    void finish_line() {
        if (!rendered_) { std::fflush(stdout); return; }
        if (gpt::log::stdout_is_tty()) {
            std::fputc('\r', stdout);
            std::fputs(line_.c_str(), stdout);
            std::fputc('\n', stdout);
        } else {
            std::fputs(line_.c_str(), stdout);
            std::fputc('\n', stdout);
        }
        std::fflush(stdout);
        rendered_ = false;
    }

    int64_t total_;
    std::string label_;
    bool enabled_;
    std::chrono::milliseconds min_interval_;
    Clock::time_point start_;
    Clock::time_point last_;
    std::string line_;
    bool finished_ = false;
    bool rendered_ = false;
};

} // namespace gpt

#endif // GPT_LOGGING_H
