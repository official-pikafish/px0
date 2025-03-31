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
#include "chess/types.h"
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
  void SetFromFen(std::string_view fen, int* rule50_ply = nullptr,
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
  BitBoard CheckersTo(const Square& ksq, const BitBoard& occupied) const;
  BitBoard RecapturesTo(const Square& sq) const;
  // Checks if "our" (white) king is under check.
  bool IsUnderCheck() const { return bool(CheckersTo(our_king_, our_pieces_ | their_pieces_).as_int()); }

  // Checks whether at least one of the sides has mating material.
  bool HasMatingMaterial() const;
  // Generates legal moves.
  MoveList GenerateLegalMoves() const;
  // Check whether pseudolegal move is legal.
  template<bool our = true>
  bool IsLegalMove(Move move) const;

  // Parses a move from move_str.
  // The input string should be in the "normal" notation rather than from the
  // player to move, i.e. "e6e5" for the black pawn move.
  // Output is currently "from the player to move" perspective (i.e. from=E3,
  // to=E4 for the same black move). This is temporary, plan is to change it
  // soon.
  Move ParseMove(std::string_view move_str) const;

  // Return a chase information in chase map
  int MakeChase(Square to) const;
  // Returns chasing information for "ours" (white)
  uint16_t UsChased() const;
  // Returns chasing information for "theirs" (black)
  uint16_t ThemChased() const;

  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), advisors_.as_int(), cannons_.as_int(),
                    pawns_.as_int(), knights_.as_int(), bishops_.as_int(),
                    __uint128_t(our_king_.as_idx() << 16 | their_king_.as_idx() << 8 | flipped_)});
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
    return BitBoard::FromSquare(our_king_) | BitBoard::FromSquare(their_king_);
  }
  bool flipped() const { return flipped_; }

  bool operator==(const ChessBoard& other) const = default;
  bool operator!=(const ChessBoard& other) const = default;

 private:
  // Sets the piece on the square.
  void PutPiece(Square square, PieceType piece, bool is_theirs);

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
  Square our_king_;
  Square their_king_;
  bool flipped_ = false;  // aka "Black to move".

  // Rule judge
  uint8_t id_board_[90];
};

}  // namespace lczero
