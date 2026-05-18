// Lightweight zero-overhead-when-off profiler.
//
// Goal: answer "where is MCTS spending its time over a whole game?", not
// just "how fast is one function?". The bench harness already measures
// raw throughput of isolated hot paths; this profiler captures the
// realistic mix — how often each function actually fires inside a full
// game played by the search.
//
// Mechanics:
//   * One global Registry keyed by string label. First hit at each call
//     site stores a Stat* in a function-local static so subsequent hits
//     are a single pointer load (no hash lookup, no mutex).
//   * Stat fields (calls, total_ns) are std::atomic so workers spawned
//     by std::async can record concurrently. The contention is one
//     atomic add per scope exit — fine for coarse-grained instrumentation
//     (functions taking ≥ ~100 ns).
//   * RAII ScopedTimer: ctor stamps start, dtor adds elapsed to the
//     stat. high_resolution_clock is steady on every platform we care
//     about (libstdc++/MSVC use QPC / clock_gettime CLOCK_MONOTONIC).
//
// Build control:
//   * Compile with -DSEQ_PROFILE to enable. SEQ_PROFILE_SCOPE expands to
//     real code; the report() / reset() entry points walk the registry.
//   * Without the define, every macro is `(void)0` and the registry stays
//     empty — same .o size, no runtime cost.
//
// Reading the output:
//   Timings are INCLUSIVE: a scope's total includes time spent inside
//   nested scopes it called. Percentages will sum to > 100 % because
//   suggest_move's time contains ismcts_iteration's time contains
//   rollout's time, etc. Subtract the inner from the outer when you
//   want exclusive ("self") time.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace seq::profile {

struct Stat {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> total_ns{0};
};

// Singleton registry of named stats. Map lookup happens once per call
// site (cached in a function-local static via the macro), then it's a
// pointer load.
class Registry {
public:
    static Registry& instance() {
        static Registry inst;
        return inst;
    }

    // Get-or-create. Thread-safe; serialized through mu_, but only the
    // first hit per label takes the lock — see the macro.
    Stat& get(const char* label) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = stats_.find(label);
        if (it != stats_.end()) return *it->second;
        auto* s = new Stat();
        stats_.emplace(label, std::unique_ptr<Stat>(s));
        order_.push_back({label, s});
        return *s;
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& kv : order_) {
            kv.stat->calls.store(0, std::memory_order_relaxed);
            kv.stat->total_ns.store(0, std::memory_order_relaxed);
        }
    }

    // Snapshot of the registry sorted by total_ns descending. Caller
    // gets a flat copy so they can iterate without holding the lock.
    struct Entry {
        std::string label;
        uint64_t    calls;
        uint64_t    total_ns;
    };
    std::vector<Entry> snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Entry> out;
        out.reserve(order_.size());
        for (auto& kv : order_) {
            out.push_back({kv.label,
                           kv.stat->calls.load(std::memory_order_relaxed),
                           kv.stat->total_ns.load(std::memory_order_relaxed)});
        }
        return out;
    }

private:
    struct OrderEntry { std::string label; Stat* stat; };
    std::mutex                                              mu_;
    std::unordered_map<std::string, std::unique_ptr<Stat>>  stats_;
    std::vector<OrderEntry>                                 order_;
};

#ifdef SEQ_PROFILE

class ScopedTimer {
public:
    explicit ScopedTimer(Stat& s) noexcept
        : stat_(s),
          t0_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() noexcept {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      t1 - t0_).count();
        stat_.calls.fetch_add(1,  std::memory_order_relaxed);
        stat_.total_ns.fetch_add(uint64_t(ns), std::memory_order_relaxed);
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Stat& stat_;
    std::chrono::high_resolution_clock::time_point t0_;
};

#define SEQ_PROFILE_CONCAT_(a, b) a##b
#define SEQ_PROFILE_CONCAT(a, b)  SEQ_PROFILE_CONCAT_(a, b)

// One macro use per scope. Resolves the Stat& once per call site via a
// function-local static (C++11 thread-safe init), then constructs an
// RAII timer. The static initializer runs exactly once per site even
// across threads, so workers calling the same instrumented function
// share one Stat and we get correct aggregate counts.
#define SEQ_PROFILE_SCOPE(label)                                              \
    static ::seq::profile::Stat& SEQ_PROFILE_CONCAT(_seq_prof_stat_, __LINE__)\
        = ::seq::profile::Registry::instance().get(label);                    \
    ::seq::profile::ScopedTimer SEQ_PROFILE_CONCAT(_seq_prof_timer_, __LINE__)\
        (SEQ_PROFILE_CONCAT(_seq_prof_stat_, __LINE__))

#else  // !SEQ_PROFILE — all macros become true no-ops.

#define SEQ_PROFILE_SCOPE(label) ((void)0)

#endif

} // namespace seq::profile
