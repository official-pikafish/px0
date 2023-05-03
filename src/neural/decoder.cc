/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "neural/decoder.h"

#include "neural/encoder.h"

namespace lczero {

namespace {

BoardSquare SingleSquare(BitBoard input) {
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return BoardSquare();
}

BitBoard MaskDiffWithMirror(const InputPlane& cur, const InputPlane& prev) {
  auto to_mirror = BitBoard(prev.mask);
  to_mirror.Mirror();
  return BitBoard(cur.mask ^ to_mirror.as_int());
}

BoardSquare OldPosition(const InputPlane& prev, BitBoard mask_diff) {
  auto to_mirror = BitBoard(prev.mask);
  to_mirror.Mirror();
  return SingleSquare(to_mirror & mask_diff);
}

}  // namespace

template <typename... T>
void MirrorAll(T&... args) {
  (..., args.Mirror());
}

void PopulateBoard(pblczero::NetworkFormat::InputFormat input_format,
                   InputPlanes planes, ChessBoard* board, int* rule50,
                   int* gameply) {
  auto rooksOurs = BitBoard(planes[0].mask);
  auto advisorsOurs = BitBoard(planes[1].mask);
  auto cannonsOurs = BitBoard(planes[2].mask);
  auto pawnsOurs = BitBoard(planes[3].mask);
  auto knightsOurs = BitBoard(planes[4].mask);
  auto bishopsOurs = BitBoard(planes[5].mask);
  auto kingOurs = BitBoard(planes[6].mask);
  auto rooksTheirs = BitBoard(planes[7].mask);
  auto advisorsTheirs = BitBoard(planes[8].mask);
  auto cannonsTheirs = BitBoard(planes[9].mask);
  auto pawnsTheirs = BitBoard(planes[10].mask);
  auto knightsTheirs = BitBoard(planes[11].mask);
  auto bishopsTheirs = BitBoard(planes[12].mask);
  auto kingTheirs = BitBoard(planes[13].mask);
  std::string fen;
  // Canonical input has no sense of side to move, so we should simply assume
  // the starting position is always white.
  bool black_to_move =
      !IsCanonicalFormat(input_format) && planes[kAuxPlaneBase].mask != 0;
  if (black_to_move) {
    // Flip to white perspective rather than side to move perspective.
    std::swap(rooksOurs, rooksTheirs);
    std::swap(advisorsOurs, advisorsTheirs);
    std::swap(cannonsOurs, cannonsTheirs);
    std::swap(pawnsOurs, pawnsTheirs);
    std::swap(knightsOurs, knightsTheirs);
    std::swap(bishopsOurs, bishopsTheirs);
    std::swap(kingOurs, kingTheirs);
    MirrorAll(rooksOurs, advisorsOurs, cannonsOurs, pawnsOurs, knightsOurs, bishopsOurs, kingOurs,
              rooksTheirs, advisorsTheirs, cannonsTheirs, pawnsTheirs, knightsTheirs, bishopsTheirs, kingTheirs);
  }
  for (int row = 9; row >= 0; --row) {
    int emptycounter = 0;
    for (int col = 0; col < 9; ++col) {
      char piece = '\0';
      if (rooksOurs.get(row, col)) {
        piece = 'R';
      } else if (rooksTheirs.get(row, col)) {
        piece = 'r';
      } else if (advisorsOurs.get(row, col)) {
        piece = 'A';
      } else if (advisorsTheirs.get(row, col)) {
        piece = 'a';
      } else if (cannonsOurs.get(row, col)) {
        piece = 'C';
      } else if (cannonsTheirs.get(row, col)) {
        piece = 'c';
      } else if (pawnsOurs.get(row, col)) {
        piece = 'P';
      } else if (pawnsTheirs.get(row, col)) {
        piece = 'p';
      } else if (knightsOurs.get(row, col)) {
        piece = 'N';
      } else if (knightsTheirs.get(row, col)) {
        piece = 'n';
      } else if (bishopsOurs.get(row, col)) {
        piece = 'B';
      } else if (bishopsTheirs.get(row, col)) {
        piece = 'b';
      } else if (kingOurs.get(row, col)) {
        piece = 'K';
      } else if (kingTheirs.get(row, col)) {
        piece = 'k';
      }
      if (emptycounter > 0 && piece) {
        fen += std::to_string(emptycounter);
        emptycounter = 0;
      }
      if (piece) {
        fen += piece;
      } else {
        emptycounter++;
      }
    }
    if (emptycounter > 0) fen += std::to_string(emptycounter);
    if (row > 0) fen += "/";
  }
  fen += " ";
  fen += black_to_move ? "b" : "w";
  fen += " - - ";
  int rule50plane = (int)planes[kAuxPlaneBase + 1].value;
  if (IsHectopliesFormat(input_format)) {
    rule50plane = (int)(120.0f * planes[kAuxPlaneBase + 1].value);
  }
  fen += std::to_string(rule50plane);
  // Reuse the 50 move rule as gameply since we don't know better.
  fen += " ";
  fen += std::to_string(rule50plane);
  board->SetFromFen(fen, rule50, gameply);
}

Move DecodeMoveFromInput(const InputPlanes& planes, const InputPlanes& prior) {
  auto rookdiff = MaskDiffWithMirror(planes[7], prior[0]);
  auto advisordiff = MaskDiffWithMirror(planes[8], prior[1]);
  auto cannondiff = MaskDiffWithMirror(planes[9], prior[2]);
  auto pawndiff = MaskDiffWithMirror(planes[10], prior[3]);
  auto knightdiff = MaskDiffWithMirror(planes[11], prior[4]);
  auto bishopdiff = MaskDiffWithMirror(planes[12], prior[5]);
  auto kingdiff = MaskDiffWithMirror(planes[13], prior[6]);
  if (rookdiff.count() == 2) {
    auto from = OldPosition(prior[0], rookdiff);
    auto to = SingleSquare(planes[7].mask & rookdiff.as_int());
    return Move(from, to);
  }
  else if (advisordiff.count() == 2) {
    auto from = OldPosition(prior[1], advisordiff);
    auto to = SingleSquare(planes[8].mask & advisordiff.as_int());
    return Move(from, to);
  }
  else if (cannondiff.count() == 2) {
    auto from = OldPosition(prior[2], cannondiff);
    auto to = SingleSquare(planes[9].mask & cannondiff.as_int());
    return Move(from, to);
  }
  else if (pawndiff.count() == 2) {
    auto from = OldPosition(prior[3], pawndiff);
    auto to = SingleSquare(planes[10].mask & pawndiff.as_int());
    return Move(from, to);
  }
  else if (knightdiff.count() == 2) {
    auto from = OldPosition(prior[4], knightdiff);
    auto to = SingleSquare(planes[11].mask & knightdiff.as_int());
    return Move(from, to);
  }
  else if (bishopdiff.count() == 2) {
    auto from = OldPosition(prior[5], bishopdiff);
    auto to = SingleSquare(planes[12].mask & bishopdiff.as_int());
    return Move(from, to);
  }
  else if (kingdiff.count() == 2) {
    auto from = OldPosition(prior[6], kingdiff);
    auto to = SingleSquare(planes[13].mask & kingdiff.as_int());
    return Move(from, to);
  }
  assert(false);
  return Move();
}

}  // namespace lczero
