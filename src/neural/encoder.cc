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

#include "neural/encoder.h"

#include <algorithm>

namespace lczero {

namespace {

int ChooseTransform(const ChessBoard& board) {
  auto our_king = uint64_t((board.kings() & board.ours()).as_int());
  int transform = NoTransform;
  if ((our_king & 0x783C1E0F0783C1E0ULL) != 0) {
    transform |= FlipTransform;
    our_king = uint64_t(FlipBoard(our_king));
  }
  // Our king is now always in left side of the palace.
  return transform;
}
}  // namespace

bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION;
}
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format) {
  return input_format ==
             pblczero::NetworkFormat::
                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == pblczero::NetworkFormat::
                             INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES;
}

int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history) {
  if (!IsCanonicalFormat(input_format)) {
    return 0;
  }
  const ChessBoard& board = history.Last().GetBoard();
  return ChooseTransform(board);
}

InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out) {
  InputPlanes result(kAuxPlaneBase + 4);

  int transform = 0;
  // Canonicalization format needs to stop early to avoid applying transform in
  // history across incompatible transitions.  It is also more canonical since
  // history before these points is not relevant to the final result.
  bool stop_early = IsCanonicalFormat(input_format);
  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    if (IsCanonicalFormat(input_format)) {
      transform = ChooseTransform(board);
    } else {
      if (we_are_black) result[kAuxPlaneBase].SetAll();
    }
    if (IsHectopliesFormat(input_format)) {
      result[kAuxPlaneBase + 1].Fill(history.Last().GetRule50Ply() / 120.0f);
    } else {
      result[kAuxPlaneBase + 1].Fill(history.Last().GetRule50Ply());
    }
    // Plane kAuxPlaneBase + 2 used to be movecount plane, now it's all zeros
    // unless we need it for canonical armageddon side to move.
    if (IsCanonicalArmageddonFormat(input_format)) {
      if (we_are_black) result[kAuxPlaneBase + 2].SetAll();
    }
    // Plane kAuxPlaneBase + 3 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 3].SetAll();
  }
  bool skip_non_repeats =
      input_format ==
          pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
      input_format == pblczero::NetworkFormat::
                          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }
    const int repetitions = position.GetRepetitions();
    // Canonical v2 only writes an item if it is a repeat, unless its the most
    // recent position.
    if (skip_non_repeats && repetitions == 0 && i > 0) {
      if (history_idx > 0) flip = !flip;
      // If no capture is 0, the previous was start of game or capture,
      // so there can't be any more repeats that are worth considering.
      if (position.GetRule50Ply() == 0) break;
      // Decrement i so it remains the same as the history_idx decrements.
      --i;
      continue;
    }

    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() & board.rooks()).as_int();
    result[base + 1].mask = (board.ours() & board.advisors()).as_int();
    result[base + 2].mask = (board.ours() & board.cannons()).as_int();
    result[base + 3].mask = (board.ours() & board.pawns()).as_int();
    result[base + 4].mask = (board.ours() & board.knights()).as_int();
    result[base + 5].mask = (board.ours() & board.bishops()).as_int();
    result[base + 6].mask = (board.ours() & board.kings()).as_int();

    result[base + 7].mask = (board.theirs() & board.rooks()).as_int();
    result[base + 8].mask = (board.theirs() & board.advisors()).as_int();
    result[base + 9].mask = (board.theirs() & board.cannons()).as_int();
    result[base + 10].mask = (board.theirs() & board.pawns()).as_int();
    result[base + 11].mask = (board.theirs() & board.knights()).as_int();
    result[base + 12].mask = (board.theirs() & board.bishops()).as_int();
    result[base + 13].mask = (board.theirs() & board.kings()).as_int();

    if (repetitions >= 1) result[base + 14].SetAll();

    if (history_idx > 0) flip = !flip;
    // If no capture is 0, the previous was start of game or capture,
    // so no need to go back further if stopping early.
    if (stop_early && position.GetRule50Ply() == 0) break;
  }
  if (transform != NoTransform) {
    // Transform all masks.
    for (int i = 0; i <= kAuxPlaneBase; i++) {
      auto v = result[i].mask;
      if (v == 0 || v == kAllSquares) continue;
      if ((transform & FlipTransform) != 0) {
        v = FlipBoard(v);
      }
      result[i].mask = v;
    }
  }
  if (transform_out) *transform_out = transform;
  return result;
}

}  // namespace lczero
