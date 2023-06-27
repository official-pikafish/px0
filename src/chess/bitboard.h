/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "utils/bititer.h"

namespace lczero {

// Stores a coordinates of a single square.
class BoardSquare {
 public:
  constexpr BoardSquare() {}
  // As a single number, 0 to 89, bottom to top, left to right.
  // 0 is a0, 9 is a1, 89 is i9.
  constexpr BoardSquare(std::uint8_t num) : square_(num) {}
  // From row(bottom to top), and col(left to right), 0-based.
  constexpr BoardSquare(int row, int col) : BoardSquare(row * 9 + col) {}
  // From Square name, e.g e4. Only lowercase.
  BoardSquare(const std::string& str, bool black = false)
      : BoardSquare(black ? '9' - str[1] : str[1] - '0', str[0] - 'a') {}
  constexpr std::uint8_t as_int() const { return square_; }
  constexpr __uint128_t as_board() const { return __uint128_t(1) << square_; }
  void set(int row, int col) { square_ = row * 9 + col; }

  BoardSquare operator+(const std::pair<int, int> directions) const {
    return BoardSquare(row() + directions.first, col() + directions.second);
  }
  BoardSquare operator-(const std::pair<int, int> directions) const {
    return BoardSquare(row() - directions.first, col() - directions.second);
  }
  BoardSquare& operator+=(const std::pair<int, int> directions) {
    set(row() + directions.first, col() + directions.second);
    return *this;
  }

  // 0-based, bottom to top.
  int row() const { return square_ / 9; }
  // 0-based, left to right.
  int col() const { return square_ % 9; }

  // Row := 9 - row.  Col remains the same.
  void Mirror() { set(9 - row(), col()); }

  // Checks whether coordinate is within 0..9.
  static bool IsValidCoordRow(int x) { return x >= 0 && x < 10; }

  // Checks whether coordinate is within 0..8.
  static bool IsValidCoordCol(int x) { return x >= 0 && x < 9; }

  // Checks whether coordinates are within 0..9 for row, within 0..8 for col.
  static bool IsValid(int row, int col) {
    return IsValidCoordRow(row) && IsValidCoordCol(col);
  }

  bool IsValid() const {
    return IsValidCoordRow(row()) && IsValidCoordCol(col());
  }

  constexpr bool operator==(const BoardSquare& other) const {
    return square_ == other.square_;
  }

  constexpr bool operator!=(const BoardSquare& other) const {
    return square_ != other.square_;
  }

  // Returns the square in algebraic notation (e.g. "e4").
  std::string as_string() const {
    return std::string(1, 'a' + col()) + std::string(1, '0' + row());
  }

 private:
  std::uint8_t square_ = 0;  // Only lower seven bits should be set.
};

// Represents a board as an array of 90 bits.
// Bit enumeration goes from bottom to top, from left to right:
// Square a0 is bit 0, square i0 is bit 8, square a1 is bit 9.
class BitBoard {
 public:
  constexpr BitBoard(__uint128_t board) : board_(board) {}
  BitBoard() = default;
  BitBoard(const BitBoard&) = default;
  BitBoard& operator=(const BitBoard&) = default;

  constexpr __uint128_t as_int() const { return board_; }
  void clear() { board_ = 0; }

  // Counts the number of set bits in the BitBoard.
  int count() const {
#if defined(NO_POPCNT)
    auto _pop_count = [&] (std::uint64_t x) {
      x -= (x >> 1) & 0x5555555555555555;
      x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
      x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
      return (x * 0x0101010101010101) >> 56;
    };
    std::uint64_t high = board_ >> 64;
    std::uint64_t low = board_;
    return _pop_count(high) + _pop_count(low);
#elif defined(_MSC_VER)
    return _mm_popcnt_u64(board_._Word[1]) + _mm_popcnt_u64(board_._Word[0]);
#else
    return __builtin_popcountll(board_ >> 64) + __builtin_popcountll(board_);
#endif
  }

  // Like count() but using algorithm faster on a very sparse BitBoard.
  // May be slower for more than 4 set bits, but still correct.
  // Useful when counting bits in a Q, R, N or B BitBoard.
  int count_few() const {
#if defined(NO_POPCNT)
    __uint128_t x = board_;
    int count;
    for (count = 0; x != 0; ++count) {
      // Clear the rightmost set bit.
      x &= x - 1;
    }
    return count;
#else
    return count();
#endif
  }

  // Sets the value for given square to 1 if cond is true.
  // Otherwise does nothing (doesn't reset!).
  void set_if(BoardSquare square, bool cond) { set_if(square.as_int(), cond); }
  void set_if(std::uint8_t pos, bool cond) {
    board_ |= (__uint128_t(cond) << pos);
  }
  void set_if(int row, int col, bool cond) {
    set_if(BoardSquare(row, col), cond);
  }

  // Sets value of given square to 1.
  void set(BoardSquare square) { set(square.as_int()); }
  void set(std::uint8_t pos) { board_ |= (__uint128_t(1) << pos); }
  void set(int row, int col) { set(BoardSquare(row, col)); }

  // Sets value of given square to 0.
  void reset(BoardSquare square) { reset(square.as_int()); }
  void reset(std::uint8_t pos) { board_ &= ~(__uint128_t(1) << pos); }
  void reset(int row, int col) { reset(BoardSquare(row, col)); }

  // Gets value of a square.
  bool get(BoardSquare square) const { return get(square.as_int()); }
  bool get(std::uint8_t pos) const {
    return bool(board_ & (__uint128_t(1) << pos));
  }
  bool get(int row, int col) const { return get(BoardSquare(row, col)); }

  // Returns whether all bits of a board are set to 0.
  bool empty() const { return board_ == 0; }

  // Checks whether two bitboards have common bits set.
  bool intersects(const BitBoard& other) const { return bool(board_ & other.board_); }

  // Flips black and white side of a board.
  void Mirror() { board_ = MirrorBoard(board_); }

  bool operator==(const BitBoard& other) const {
    return board_ == other.board_;
  }

  bool operator!=(const BitBoard& other) const {
    return board_ != other.board_;
  }

  BitIterator<BoardSquare> begin() const { return board_; }
  BitIterator<BoardSquare> end() const { return __uint128_t(0); }

  std::string DebugString() const {
    std::string res;
    for (int i = 9; i >= 0; --i) {
      for (int j = 0; j < 9; ++j) {
        if (get(i, j))
          res += '#';
        else
          res += '.';
      }
      res += '\n';
    }
    return res;
  }

  // Applies a mask to the bitboard (intersects).
  BitBoard& operator&=(const BitBoard& a) {
    board_ &= a.board_;
    return *this;
  }

  BitBoard& operator-=(const BitBoard& a) {
    board_ &= ~a.board_;
    return *this;
  }

  BitBoard& operator|=(const BitBoard& a) {
    board_ |= a.board_;
    return *this;
  }

  friend void swap(BitBoard& a, BitBoard& b) {
    using std::swap;
    swap(a.board_, b.board_);
  }

  // Returns union (bitwise OR) of two boards.
  friend BitBoard operator|(const BitBoard& a, const BitBoard& b) {
    return {a.board_ | b.board_};
  }

  // Returns intersection (bitwise AND) of two boards.
  friend BitBoard operator&(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & b.board_};
  }

  // Returns bitboard with one bit reset.
  friend BitBoard operator-(const BitBoard& a, const BoardSquare& b) {
    return {a.board_ & ~b.as_board()};
  }

  // Returns difference (bitwise AND-NOT) of two boards.
  friend BitBoard operator-(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & ~b.board_};
  }

 private:
  __uint128_t board_ = 0;
};

class Move {
 public:
  Move() = default;
  constexpr Move(BoardSquare from, BoardSquare to)
      : data_(to.as_int() + (from.as_int() << 7)) {}
  Move(const std::string& str, bool black = false);
  Move(const char* str, bool black = false) : Move(std::string(str), black) {}

  BoardSquare to() const { return BoardSquare(data_ & kToMask); }
  BoardSquare from() const { return BoardSquare((data_ & kFromMask) >> 7); }

  void SetTo(BoardSquare to) { data_ = (data_ & ~kToMask) | to.as_int(); }
  void SetFrom(BoardSquare from) {
    data_ = (data_ & ~kFromMask) | (from.as_int() << 7);
  }
  // 0 .. 16384.
  uint16_t as_packed_int() const;

  // 0 .. 2061, to use in neural networks.
  // Transform is a bit field which describes a transform to be applied to the
  // the move before converting it to an index.
  uint16_t as_nn_index(int transform) const;

  explicit operator bool() const { return data_ != 0; }
  bool operator==(const Move& other) const { return data_ == other.data_; }

  void Mirror() {
    BoardSquare f = from();
    f.Mirror();
    SetFrom(f);
    BoardSquare t = to();
    t.Mirror();
    SetTo(t);
  }

  std::string as_string() const {
    return from().as_string() + to().as_string();
  }

 private:
  uint16_t data_ = 0;
  // Move, using the following encoding:
  // bits 0..6 "to"-square
  // bits 7..13 "from"-square

  enum Masks : uint16_t {
    kToMask = 0b0000000001111111,
    kFromMask = 0b0011111110000000,
  };
};

using MoveList = std::vector<Move>;

// Gets the move from the NN move index, undoing the given transform.
Move MoveFromNNIndex(int idx, int transform);

}  // namespace lczero
