// xoshiro256++ — Blackman & Vigna 2018, public domain.
//
// Reasons we use it instead of std::mt19937_64:
//   - state is 32 bytes (vs ~2500 for mt19937), so copies during MCTS
//     tree-node expansion are essentially free
//   - generation is ~3x faster (a handful of 64-bit ops, no division,
//     no large-array indexing)
//   - passes BigCrush and PractRand; statistical quality is fine for
//     game-playing simulations
//
// Implements the UniformRandomBitGenerator named requirement, so it
// works directly with std::shuffle and std::uniform_int_distribution.

#pragma once
#include <cstdint>

namespace seq {

class Xoshiro256pp {
public:
    using result_type = uint64_t;

    Xoshiro256pp()                   { seed(0xC0FFEE12345ULL); }
    explicit Xoshiro256pp(uint64_t s) { seed(s); }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~uint64_t(0); }

    // splitmix64 from the same authors — used to expand a single 64-bit
    // seed into the four 64-bit words xoshiro needs. Important for
    // avoiding poor early-output behavior when seeded with e.g. 0.
    void seed(uint64_t s) {
        for (int i = 0; i < 4; ++i) {
            s += 0x9E3779B97F4A7C15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            s_[i] = z;
        }
    }

    result_type operator()() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Convenience helpers used a lot in the search code.
    int    rand_int(int lo, int hi_inclusive) {
        // Unbiased only up to constant-factor; bias is irrelevant at the
        // ranges we use (move list sizes < 256).
        uint64_t range = uint64_t(hi_inclusive - lo) + 1;
        return lo + int((*this)() % range);
    }
    double rand_double() {
        // 53-bit mantissa worth of bits, mapped to [0, 1).
        return ((*this)() >> 11) * (1.0 / (uint64_t(1) << 53));
    }

private:
    static constexpr uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    uint64_t s_[4];
};

} // namespace seq
