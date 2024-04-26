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
#include <string>

#include "chess/bitboard.h"
#include "utils/hashcat.h"

namespace lczero {

// Initializes internal magic bitboard structures.
void InitializeMagicBitboards();

// Represents a board position.
// Unlike most chess engines, the board is mirrored for black.
class ChessBoard {
 public:
  ChessBoard() = default;
  ChessBoard(const ChessBoard&) = default;
  ChessBoard(const std::string& fen) { SetFromFen(fen); }

  ChessBoard& operator=(const ChessBoard&) = default;

  static const char* kStartposFen;
  static const ChessBoard kStartposBoard;
  static const BitBoard kPawnMask;

  // Sets position from FEN string.
  // If @rule50_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(std::string fen, int* rule50_ply = nullptr,
                  int* moves = nullptr);
  // Nullifies the whole structure.
  void Clear();
  // Swaps black and white pieces and mirrors them relative to the
  // middle of the board. (what was on rank 0 appears on rank 9, what was
  // on file b remains on file b).
  void Mirror();

  // Generates list of possible moves for "ours" (white), but may leave king
  // under check.
  MoveList GeneratePseudolegalMoves() const;
  // Applies the move. (Only for "ours" (white)). Returns true if 50 moves
  // counter should be removed.
  bool ApplyMove(Move move);
  // Checkers BitBoard to square sq
  template<bool our = true>
  BitBoard CheckersTo(const BoardSquare &ksq, const BitBoard& occupied) const;
  BitBoard RecapturesTo(const BoardSquare &sq) const;
  // Checks if "our" (white) king is under check.
  bool IsUnderCheck() const { return bool(CheckersTo(our_king_, our_pieces_ | their_pieces_).as_int()); }

  // Checks whether at least one of the sides has mating material.
  bool HasMatingMaterial() const;
  // Generates legal moves.
  MoveList GenerateLegalMoves() const;
  // Check whether pseudolegal move is legal.
  template<bool our = true>
  bool IsLegalMove(Move move) const;
  // Returns whether two moves are actually the same move in the position.
  bool IsSameMove(Move move1, Move move2) const;
  // Return a chase information in chase map
  int MakeChase(BoardSquare to) const;
  // Returns chasing information for "ours" (white)
  uint16_t Chased() const;

  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), advisors_.as_int(), cannons_.as_int(),
                    pawns_.as_int(), knights_.as_int(), bishops_.as_int(),
                    __uint128_t(our_king_.as_int() << 16 | their_king_.as_int() << 8 | flipped_)});
  }

  std::string DebugString() const;

  BitBoard ours() const { return our_pieces_; }
  BitBoard theirs() const { return their_pieces_; }
  BitBoard rooks() const { return rooks_; }
  BitBoard advisors() const { return advisors_; }
  BitBoard cannons() const { return cannons_; }
  BitBoard pawns() const { return pawns_; }
  BitBoard knights() const { return knights_; }
  BitBoard bishops() const { return bishops_; }
  BitBoard kings() const {
    return our_king_.as_board() | their_king_.as_board();
  }
  bool flipped() const { return flipped_; }

  bool operator==(const ChessBoard& other) const {
    return (our_pieces_ == other.our_pieces_) && (their_pieces_ == other.their_pieces_) &&
           (rooks_ == other.rooks_) && (advisors_ == other.advisors_) &&
           (cannons_ == other.cannons_) && (pawns_ == other.pawns_) &&
           (knights_ == other.knights_) && (bishops_ == other.bishops_) &&
           (our_king_ == other.our_king_) && (their_king_ == other.their_king_) &&
           (flipped_ == other.flipped_);
  }

  bool operator!=(const ChessBoard& other) const { return !operator==(other); }

  enum Square : uint8_t {
    // clang-format off
    A0 = 0, B0, C0, D0, E0, F0, G0, H0, I0,
    A1, B1, C1, D1, E1, F1, G1, H1, I1,
    A2, B2, C2, D2, E2, F2, G2, H2, I2,
    A3, B3, C3, D3, E3, F3, G3, H3, I3,
    A4, B4, C4, D4, E4, F4, G4, H4, I4,
    A5, B5, C5, D5, E5, F5, G5, H5, I5,
    A6, B6, C6, D6, E6, F6, G6, H6, I6,
    A7, B7, C7, D7, E7, F7, G7, H7, I7,
    A8, B8, C8, D8, E8, F8, G8, H8, I8,
    A9, B9, C9, D9, E9, F9, G9, H9, I9,
    // clang-format on
  };

  enum File : uint8_t {
    // clang-format off
    FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I, FILE_NB
    // clang-format on
  };

  enum Rank : uint8_t {
    // clang-format off
    RANK_0 = 0, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9, RANK_NB
    // clang-format on
  };

  enum PieceType : uint8_t {
    // clang-format off
    ROOK, ADVISOR, CANNON, PAWN, KNIGHT, BISHOP, KING, KNIGHT_TO, PAWN_TO_OURS, PAWN_TO_THEIRS, PIECE_TYPE_NB
    // clang-format on
  };

 private:
  // All white pieces.
  BitBoard our_pieces_;
  // All black pieces.
  BitBoard their_pieces_;
  // Rooks.
  BitBoard rooks_;
  // Advisors.
  BitBoard advisors_;
  // Cannons.
  BitBoard cannons_;
  // Pawns.
  BitBoard pawns_;
  // Knights;
  BitBoard knights_;
  // Bishops;
  BitBoard bishops_;
  BoardSquare our_king_;
  BoardSquare their_king_;
  bool flipped_ = false;  // aka "Black to move".

  // Rule judge
  uint8_t id_board_[90];
};

}  // namespace lczero
