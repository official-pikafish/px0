/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once
#include <cstdint>
#include <iterator>
#ifdef _MSC_VER
#include <intrin.h>
#include <__msvc_int128.hpp>
using __uint128_t = std::_Unsigned128;
#endif

namespace lczero {

inline unsigned long GetLowestBit(__uint128_t value) {
#if defined(_MSC_VER) // MSVC
  unsigned long idx;
  if (value._Word[0])
  {
    _BitScanForward64(&idx, value._Word[0]);
    return idx;
  }
  else
  {
    _BitScanForward64(&idx, value._Word[1]);
    return idx + 64;
  }
#else // Assumed gcc or compatible compiler
  if (uint64_t(value))
    return __builtin_ctzll(value);
  return __builtin_ctzll(value >> 64) + 64;
#endif
}

enum BoardTransform {
  NoTransform = 0,
  // Horizontal mirror
  FlipTransform = 1,
};

inline __uint128_t FlipBoard(__uint128_t v) {
  constexpr __uint128_t seq1 = __uint128_t(0x0000000000201008ULL) << 64 | 0x0402010080402010ULL;
  constexpr __uint128_t seq2 = __uint128_t(0x0000000003C1E0F0ULL) << 64 | 0x783C1E0F0783C1E0ULL;
  constexpr __uint128_t seq3 = __uint128_t(0x0000000003198CC6ULL) << 64 | 0x633198CC6633198CULL;
  constexpr __uint128_t seq4 = __uint128_t(0x0000000002954AA5ULL) << 64 | 0x52A954AA552A954AULL;

  __uint128_t fixed = v & seq1;
  v = ((v & seq2) >> 5) | ((v << 5) & seq2);
  v = ((v & seq3) >> 2) | ((v << 2) & seq3);
  v = ((v & seq4) >> 1) | ((v << 1) & seq4);
  return v | fixed;
}

inline __uint128_t MirrorBoard(__uint128_t v) {
  constexpr __uint128_t seq1 = __uint128_t(0x0000000000000000ULL) << 64 | 0x00001FFFFFFFFFFFULL;
  constexpr __uint128_t seq2 = __uint128_t(0x00000000000000FFULL) << 64 | 0x8000000007FC0000ULL;
  constexpr __uint128_t seq3 = __uint128_t(0x0000000000000000ULL) << 64 | 0x7FFFE0000003FFFFULL;
  constexpr __uint128_t seq4 = __uint128_t(0x000000000001FF00ULL) << 64 | 0x003FE00FF80001FFULL;

  v = ((v & seq1) << 45) | ((v >> 45) & seq1);
  __uint128_t fixed = v & seq2;
  v = ((v & seq3) << 27) | ((v >> 27) & seq3);
  v = ((v & seq4) <<  9) | ((v >>  9) & seq4);
  return v | fixed;
}

// Iterates over all set bits of the value, lower to upper. The value of
// dereferenced iterator is bit number (lower to upper, 0 bazed)
template <typename T>
class BitIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = T;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  BitIterator(__uint128_t value) : value_(value){};
  bool operator!=(const BitIterator& other) { return value_ != other.value_; }

  void operator++() { value_ &= (value_ - 1); }
  T operator*() const { return GetLowestBit(value_); }

 private:
  __uint128_t value_;
};

class IterateBits {
 public:
  IterateBits(__uint128_t value) : value_(value) {}
  BitIterator<int> begin() { return value_; }
  BitIterator<int> end() { return __uint128_t(0); }

 private:
  __uint128_t value_;
};

}  // namespace lczero
