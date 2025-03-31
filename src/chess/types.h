/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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
#include <string>
#include <vector>

namespace lczero {

struct PieceType {
  uint8_t idx;
  static constexpr PieceType FromIdx(uint8_t idx) { return PieceType{idx}; }
  static PieceType Parse(char c);
  std::string ToString(bool uppercase = false) const {
    return std::string(1, "racpnbk"[idx] + (uppercase ? 'A' - 'a' : 0));
  }
  bool IsValid() const { return idx < 7; }
  bool operator==(const PieceType& other) const = default;
  bool operator!=(const PieceType& other) const = default;

 private:
  constexpr explicit PieceType(uint8_t idx) : idx(idx) {}
};

constexpr PieceType kRook = PieceType::FromIdx(0),
                    kAdvisor = PieceType::FromIdx(1),
                    kCannon = PieceType::FromIdx(2),
                    kPawn = PieceType::FromIdx(3),
                    kKnight = PieceType::FromIdx(4),
                    kBishop = PieceType::FromIdx(5),
                    kKing = PieceType::FromIdx(6),
                    kPieceTypeNB = PieceType::FromIdx(7),
                    kKnightTo = PieceType::FromIdx(7),
                    kPawnToOurs = PieceType::FromIdx(8),
                    kPawnToTheirs = PieceType::FromIdx(9);

struct File {
  uint8_t idx;
  File() : idx(0x80) {}  // Not on board.
  constexpr bool IsValid() const { return idx < 9; }
  static constexpr File FromIdx(uint8_t idx) { return File{idx}; }
  static constexpr File Parse(char c) { return File(std::tolower(c) - 'a'); }
  std::string ToString(bool uppercase = false) const {
    return std::string(1, (uppercase ? 'A' : 'a') + idx);
  }
  void Flop() { idx = 8 - idx; }
  auto operator<=>(const File& other) const = default;
  void operator++() { ++idx; }
  void operator--() { --idx; }
  void operator+=(int delta) { idx += delta; }
  File operator+(int delta) const { return File(idx + delta); }
  File operator-(int delta) const { return File(idx - delta); }

 private:
  constexpr explicit File(uint8_t idx) : idx(idx) {}
};

constexpr File kFileA = File::FromIdx(0), kFileB = File::FromIdx(1),
               kFileC = File::FromIdx(2), kFileD = File::FromIdx(3),
               kFileE = File::FromIdx(4), kFileF = File::FromIdx(5),
               kFileG = File::FromIdx(6), kFileH = File::FromIdx(7),
               kFileI = File::FromIdx(8), kFileNB = File::FromIdx(9);

struct Rank {
  uint8_t idx;
  constexpr bool IsValid() const { return idx < 10; }
  static constexpr Rank FromIdx(uint8_t idx) { return Rank{idx}; }
  static constexpr Rank Parse(char c) { return Rank(c - '0'); }
  void Flip() { idx = 9 - idx; }
  std::string ToString() const { return std::string(1, '0' + idx); }
  auto operator<=>(const Rank& other) const = default;
  void operator--() { --idx; }
  void operator++() { ++idx; }
  void operator+=(int delta) { idx += delta; }
  Rank operator+(int delta) const { return Rank(idx + delta); }
  Rank operator-(int delta) const { return Rank(idx - delta); }

 private:
  constexpr explicit Rank(uint8_t idx) : idx(idx) {}
};

constexpr Rank kRank0 = Rank::FromIdx(0), kRank1 = Rank::FromIdx(1),
               kRank2 = Rank::FromIdx(2), kRank3 = Rank::FromIdx(3),
               kRank4 = Rank::FromIdx(4), kRank5 = Rank::FromIdx(5),
               kRank6 = Rank::FromIdx(6), kRank7 = Rank::FromIdx(7),
               kRank8 = Rank::FromIdx(8), kRank9 = Rank::FromIdx(9),
               kRankNB = Rank::FromIdx(10);

// Stores a coordinates of a single square.
class Square {
 public:
  constexpr Square() = default;
  constexpr Square(File file, Rank rank) : idx_(rank.idx * kFileNB.idx + file.idx) {}
  static constexpr Square FromIdx(uint8_t idx) { return Square{idx}; }
  static constexpr Square Parse(std::string_view);
  constexpr File file() const { return File::FromIdx(idx_ % kFileNB.idx); }
  constexpr Rank rank() const { return Rank::FromIdx(idx_ / kFileNB.idx); }
  // Flips the ranks. 1 becomes 8, 2 becomes 7, etc. Files remain the same.
  void Flip() { 
    Rank r = rank();
    r.Flip();
    idx_ = r.idx * kFileNB.idx + file().idx;
  }
  std::string ToString(bool uppercase = false) const {
    return file().ToString(uppercase) + rank().ToString();
  }
  constexpr bool operator==(const Square& other) const = default;
  constexpr bool operator!=(const Square& other) const = default;
  Square operator+(const std::pair<int, int> directions) const {
    return Square(file() + directions.second, rank() + directions.first);
  }
  Square operator-(const std::pair<int, int> directions) const {
    return Square(file() - directions.second, rank() - directions.first);
  }
  Square& operator+=(const std::pair<int, int> directions) {
    idx_ = Square(file() + directions.second, rank() + directions.first).idx_;
    return *this;
  }
  constexpr bool IsValid() const { return file().IsValid() && rank().IsValid(); }
  constexpr uint8_t as_idx() const { return idx_; }

 private:
  explicit constexpr Square(uint8_t idx) : idx_(idx) {}

  // 0 is a0, 1 is b0, 9 is a1, 89 is h9.
  uint8_t idx_;
};

constexpr Square kSquareA0 = Square(kFileA, kRank0),
                 kSquareC0 = Square(kFileC, kRank0),
                 kSquareE0 = Square(kFileE, kRank0),
                 kSquareG0 = Square(kFileG, kRank0),
                 kSquareH0 = Square(kFileH, kRank0);

class Move {
 public:
  Move() = default;
  static constexpr Move White(Square from, Square to) {
    return Move((from.as_idx() << 7) | to.as_idx());
  }

  bool operator==(const Move& other) const = default;
  bool operator!=(const Move& other) const = default;

  // Mirrors the ranks of the move.
  void Flip() {
    Square f = from();
    Square t = to();
    f.Flip();
    t.Flip();
    data_ = f.as_idx() << 7 | t.as_idx();
  }
  std::string ToString() const;

  Square from() const { return Square::FromIdx((data_ & kFromMask) >> 7); }
  Square to() const { return Square::FromIdx(data_ & kToMask); }
  // TODO remove this once UciReponder starts using std::optional for ponder.
  bool is_null() const { return data_ == 0; }

  uint16_t raw_data() const { return data_; }

 private:
  explicit constexpr Move(uint16_t data) : data_(data) {}

  // Move encoding using 16 bits:
  // - bits  0-6:  "to" square (7 bits)
  // - bits  7-13: "from" square (7 bits)
  // - bit   14-15:   reserved (potentially for side-to-move)
  uint16_t data_ = 0;

  enum Masks : uint16_t {
    // clang-format off
    kToMask = 0b0000000001111111,
    kFromMask = 0b0011111110000000,
    // clang-format on
  };
};

inline int operator-(File a, File b) { return static_cast<int>(a.idx) - b.idx; }
inline int operator-(Rank a, Rank b) { return static_cast<int>(a.idx) - b.idx; }

inline constexpr Square Square::Parse(std::string_view str) {
  return Square(File::Parse(str[0]), Rank::Parse(str[1]));
}

inline PieceType PieceType::Parse(char c) {
  switch (tolower(c)) {
    case 'r':
      return kRook;
    case 'a':
      return kAdvisor;
    case 'c':
      return kCannon;
    case 'p':
      return kPawn;
    case 'n':
      return kKnight;
    case 'b':
      return kBishop;
    case 'k':
      return kKing;
    default:
      return PieceType{7};
  }
}

inline std::string Move::ToString() const {
  return from().ToString() + to().ToString();
}

using MoveList = std::vector<Move>;

}  // namespace lczero