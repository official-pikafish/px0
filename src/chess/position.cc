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

#include "chess/position.h"
#include "utils/exception.h"

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "chess/types.h"

namespace lczero {
namespace {
// GetPieceAt returns the piece found at row, col on board or the null-char '\0'
// in case no piece there.
char GetPieceAt(const lczero::ChessBoard& board, int row, int col) {
  char c = '\0';
  const Square square(File::FromIdx(col), Rank::FromIdx(row));
  if (board.ours().get(square) || board.theirs().get(square)) {
    if (board.rooks().get(square)) {
      c = 'R';
    } else if (board.advisors().get(square)) {
      c = 'A';
    } else if (board.cannons().get(square)) {
      c = 'C';
    } else if (board.pawns().get(square)) {
      c = 'P';
    } else if (board.knights().get(square)) {
      c = 'N';
    } else if (board.bishops().get(square)) {
      c = 'B';
    } else if (board.kings().get(square)) {
      c = 'K';
    }
    if (board.theirs().get(square)) {
      c = std::tolower(c);  // Capitals are for white.
    }
  }
  return c;
}
}  // namespace

Position::Position(const Position& parent, Move m)
    : us_board_(parent.us_board_),
      rule50_ply_(parent.rule50_ply_),
      us_check(parent.them_check),
      them_check(parent.us_check),
      ply_count_(parent.ply_count_ + 1) {
  const bool is_zeroing = us_board_.ApplyMove(m);
  us_board_.Mirror();
  if (!us_board_.IsUnderCheck() || ++them_check <= 10) {
    if (us_check > 10 && parent.us_board_.IsUnderCheck())
      ++us_check;
    else
      ++rule50_ply_;
  }
  if (is_zeroing) {
    rule50_ply_ = 0;
    us_check = 0;
    them_check = 0;
  }
}

Position::Position(const ChessBoard& board, int rule50_ply, int game_ply)
    : rule50_ply_(rule50_ply), repetitions_(0), ply_count_(game_ply) {
  us_board_ = board;
}

Position Position::FromFen(std::string_view fen) {
  Position pos;
  pos.us_board_.SetFromFen(std::string(fen), &pos.rule50_ply_, &pos.ply_count_);
  return pos;
}

uint64_t Position::Hash() const {
  return HashCat({us_board_.Hash(), static_cast<unsigned long>(repetitions_)});
}

std::string Position::DebugString() const { return us_board_.DebugString(); }

GameResult operator-(const GameResult& res) {
  return res == GameResult::BLACK_WON   ? GameResult::WHITE_WON
         : res == GameResult::WHITE_WON ? GameResult::BLACK_WON
                                        : res;
}

GameResult PositionHistory::ComputeGameResult() const {
  const auto& board = Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    return IsBlackToMove() ? GameResult::WHITE_WON : GameResult::BLACK_WON;
  }

  if (Last().GetRepetitions() >= 2) {
    GameResult result = RuleJudge();
    return IsBlackToMove() ? result : -result;
  }
  if (!board.HasMatingMaterial()) return GameResult::DRAW;
  if (Last().GetRule50Ply() >= 120) return GameResult::DRAW;

  return GameResult::UNDECIDED;
}

void PositionHistory::Reset(const ChessBoard& board, int rule50_ply,
                            int game_ply) {
  positions_.clear();
  positions_.emplace_back(board, rule50_ply, game_ply);
}

void PositionHistory::Reset(const Position& pos) {
  positions_.clear();
  positions_.push_back(pos);
}

void PositionHistory::Append(Move m) {
  // TODO(mooskagh) That should be emplace_back(Last(), m), but MSVS STL
  //                has a bug in implementation of emplace_back, when
  //                reallocation happens. (it also reallocates Last())
  positions_.push_back(Position(Last(), m));
  int cycle_length;
  int repetitions = ComputeLastMoveRepetitions(&cycle_length);
  positions_.back().SetRepetitions(repetitions, cycle_length);
}

GameResult PositionHistory::RuleJudge() const {
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetRule50Ply() < 4) return GameResult::UNDECIDED;

  bool checkThem = last.GetBoard().IsUnderCheck();
  bool checkUs = positions_[size(positions_) - 2].GetBoard().IsUnderCheck();
  uint16_t chaseThem = last.GetBoard().ThemChased() &
                       ~positions_[size(positions_) - 2].GetBoard().UsChased();
  uint16_t chaseUs = positions_[size(positions_) - 2].GetBoard().ThemChased() &
                     ~positions_[size(positions_) - 3].GetBoard().UsChased();

  for (int idx = positions_.size() - 3; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard().IsUnderCheck())
      chaseThem = chaseUs = 0;
    else
      checkThem = false;

    if (pos.GetBoard() == last.GetBoard() && pos.GetRepetitions() == 0) {
      return (checkThem || checkUs)   ? (!checkUs     ? GameResult::BLACK_WON
                                         : !checkThem ? GameResult::WHITE_WON
                                                      : GameResult::DRAW)
             : (chaseThem || chaseUs) ? (!chaseUs     ? GameResult::BLACK_WON
                                         : !chaseThem ? GameResult::WHITE_WON
                                                      : GameResult::DRAW)
                                      : GameResult::DRAW;
    }

    if (idx - 1 >= 0) {
      if (positions_[idx - 1].GetBoard().IsUnderCheck())
        chaseThem = chaseUs = 0;
      else
        checkUs = false;
      chaseThem &= pos.GetBoard().ThemChased() &
                   ~positions_[idx - 1].GetBoard().UsChased();
      if (idx - 2 >= 0)
        chaseUs &= positions_[idx - 1].GetBoard().ThemChased() &
                   ~positions_[idx - 2].GetBoard().UsChased();
    }
  }

  throw Exception("Judging non-repetition move sequence");
}

int PositionHistory::ComputeLastMoveRepetitions(int* cycle_length) const {
  *cycle_length = 0;
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetRule50Ply() < 4) return 0;

  for (int idx = positions_.size() - 5; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      *cycle_length = positions_.size() - 1 - idx;
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetRule50Ply() < 2) return 0;
  }
  return 0;
}

bool PositionHistory::DidRepeatSinceLastZeroingMove() const {
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (iter->GetRepetitions() > 0) return true;
    if (iter->GetRule50Ply() == 0) return false;
  }
  return false;
}

uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetRule50Ply());
}

std::string GetFen(const Position& pos) {
  std::string result;
  ChessBoard board = pos.GetBoard();
  if (board.flipped()) board.Mirror();
  for (int row = 9; row >= 0; --row) {
    int emptycounter = 0;
    for (int col = 0; col < 9; ++col) {
      char piece = GetPieceAt(board, row, col);
      if (emptycounter > 0 && piece) {
        result += std::to_string(emptycounter);
        emptycounter = 0;
      }
      if (piece) {
        result += piece;
      } else {
        emptycounter++;
      }
    }
    if (emptycounter > 0) result += std::to_string(emptycounter);
    if (row > 0) result += "/";
  }
  result += pos.IsBlackToMove() ? " b" : " w";
  result += " - - " + std::to_string(pos.GetRule50Ply());
  result += " " + std::to_string(
                      (pos.GetGamePly() + (pos.IsBlackToMove() ? 1 : 2)) / 2);
  return result;
}
}  // namespace lczero
