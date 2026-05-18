// Core types for the Sequence engine.
//
// The 10x10 board is 100 cells, which doesn't fit in a single uint64_t,
// so we use a pair of them. cell index = row * 10 + col, 0..99.
//   bits  0..63 live in `lo` (cells  0..63)
//   bits  0..35 live in `hi` (cells 64..99)
// `hi`'s upper 28 bits must stay zero — enforced by HI_MASK in operator~.

#pragma once
#include <cstdint>
#include <cstddef>

namespace seq {

constexpr int N_CELLS = 100;
constexpr int N_ROWS = 10;
constexpr int N_COLS = 10;
constexpr int HAND_SIZE = 7;
constexpr int SEQUENCE_LENGTH = 5;
constexpr int SEQUENCES_TO_WIN = 2;

// Chip states stored per cell. EMPTY/PLAYER0/etc. mirror the Python engine.
enum Chip : int8_t {
    EMPTY     = 0,
    PLAYER0   = 1,
    PLAYER1   = 2,
    SEQUENCE0 = 3,   // locked sequence chip for player 0
    SEQUENCE1 = 4,
};

inline constexpr int cell_of(int r, int c) { return r * N_COLS + c; }
inline constexpr int row_of(int cell)      { return cell / N_COLS; }
inline constexpr int col_of(int cell)      { return cell % N_COLS; }

// 100-bit bitboard built from two uint64s. Designed to be trivially
// copyable and small enough to live in registers across calls.
struct BB100 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    static constexpr uint64_t HI_MASK = (uint64_t(1) << 36) - 1;

    constexpr bool test(int i) const {
        return i < 64 ? ((lo >> i) & 1ULL) != 0
                      : ((hi >> (i - 64)) & 1ULL) != 0;
    }
    constexpr void set(int i) {
        if (i < 64) lo |= (uint64_t(1) << i);
        else        hi |= (uint64_t(1) << (i - 64));
    }
    constexpr void reset(int i) {
        if (i < 64) lo &= ~(uint64_t(1) << i);
        else        hi &= ~(uint64_t(1) << (i - 64));
    }
    constexpr int popcount() const {
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
    }
    constexpr bool empty() const { return lo == 0 && hi == 0; }
    constexpr bool any()   const { return lo != 0 || hi != 0; }

    constexpr BB100 operator|(BB100 o) const { return {lo | o.lo, hi | o.hi}; }
    constexpr BB100 operator&(BB100 o) const { return {lo & o.lo, hi & o.hi}; }
    constexpr BB100 operator^(BB100 o) const { return {lo ^ o.lo, hi ^ o.hi}; }
    constexpr BB100 operator~() const { return {~lo, (~hi) & HI_MASK}; }

    constexpr BB100& operator|=(BB100 o) { lo |= o.lo; hi |= o.hi; return *this; }
    constexpr BB100& operator&=(BB100 o) { lo &= o.lo; hi &= o.hi; return *this; }
    constexpr BB100& operator^=(BB100 o) { lo ^= o.lo; hi ^= o.hi; return *this; }

    constexpr bool operator==(BB100 o) const { return lo == o.lo && hi == o.hi; }
    constexpr bool operator!=(BB100 o) const { return lo != o.lo || hi != o.hi; }

    // Test "are all bits in `subset` set in this bitboard"
    constexpr bool contains(BB100 subset) const {
        return (lo & subset.lo) == subset.lo && (hi & subset.hi) == subset.hi;
    }

    // Iterate set bits with `while (bb.any()) { int i = bb.pop_lsb(); ... }`
    int pop_lsb() {
        if (lo) {
            int i = __builtin_ctzll(lo);
            lo &= lo - 1;
            return i;
        }
        int i = __builtin_ctzll(hi);
        hi &= hi - 1;
        return i + 64;
    }
};

inline BB100 bit(int i) {
    BB100 b;
    b.set(i);
    return b;
}

} // namespace seq
