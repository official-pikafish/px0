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
*/

#include "src/neural/encoder.h"

#include <gtest/gtest.h>

namespace lczero {

TEST(EncodePositionForNN, EncodeStartPosition) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  InputPlane our_rooks_plane = encoded_planes[0];
  EXPECT_EQ(our_rooks_plane.mask, 1ull | (1ull << 8));
  EXPECT_EQ(our_rooks_plane.value, 1.0f);

  InputPlane our_advisors_plane = encoded_planes[1];
  EXPECT_EQ(our_advisors_plane.mask, (1ull << 3) | (1ull << 5));
  EXPECT_EQ(our_advisors_plane.value, 1.0f);

  InputPlane our_cannons_plane = encoded_planes[2];
  EXPECT_EQ(our_cannons_plane.mask, (1ull << 19) | (1ull << 25));
  EXPECT_EQ(our_cannons_plane.value, 1.0f);

  InputPlane our_pawns_plane = encoded_planes[3];
  auto our_pawns_mask = 0ull;
  for (auto i = 0; i < 10; i += 2) {
    // First pawn is at square a3 (position 27)
    // Last pawn is at square i3 (position 27 + 8 = 35)
    our_pawns_mask |= __uint128_t(1) << (27 + i);
  }
  EXPECT_EQ(our_pawns_plane.mask, our_pawns_mask);
  EXPECT_EQ(our_pawns_plane.value, 1.0f);

  InputPlane our_knights_plane = encoded_planes[4];
  EXPECT_EQ(our_knights_plane.mask, (1ull << 1) | (1ull << 7));
  EXPECT_EQ(our_knights_plane.value, 1.0f);

  InputPlane our_bishops_plane = encoded_planes[5];
  EXPECT_EQ(our_bishops_plane.mask, (1ull << 2) | (1ull << 6));
  EXPECT_EQ(our_bishops_plane.value, 1.0f);

  InputPlane our_king_plane = encoded_planes[6];
  EXPECT_EQ(our_king_plane.mask, 1ull << 4);
  EXPECT_EQ(our_king_plane.value, 1.0f);

  // Sanity check opponent's pieces
  InputPlane their_king_plane = encoded_planes[13];
  auto their_king_row = 9;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane.mask,
            __uint128_t(1) << (9 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane.value, 1.0f);

  // Start of game, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 15; j++) {
      InputPlane zeroed_history = encoded_planes[15 + i * 15 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Auxiliary planes

  InputPlane we_are_black_plane = encoded_planes[15 * 8];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[15 * 8 + 1];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquares);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[15 * 8 + 2];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[15 * 8 + 3];
  EXPECT_EQ(all_ones_plane.mask, kAllSquares);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeFiftyMoveCounter) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 1. h2e2
  history.Append(Move("h2e2", false));

  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  InputPlane we_are_black_plane = encoded_planes[15 * 8];
  EXPECT_EQ(we_are_black_plane.mask, kAllSquares);
  EXPECT_EQ(we_are_black_plane.value, 1.0f);

  InputPlane fifty_move_counter_plane = encoded_planes[15 * 8 + 1];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquares);
  EXPECT_EQ(fifty_move_counter_plane.value, 1.0f);

  // 1. h2e2 h9g7
  history.Append(Move("h9g7", true));

  encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  we_are_black_plane = encoded_planes[15 * 8];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  fifty_move_counter_plane = encoded_planes[15 * 8 + 1];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquares);
  EXPECT_EQ(fifty_move_counter_plane.value, 2.0f);
}

TEST(EncodePositionForNN, EncodeFiftyMoveCounterFormat2) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 1. h2e2
  history.Append(Move("h2e2", false));

  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  InputPlane zerod_plane = encoded_planes[15 * 8];
  EXPECT_EQ(zerod_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[15 * 8 + 1];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquares);
  EXPECT_EQ(fifty_move_counter_plane.value, 1.0f);

  // 1. h2e2 h9g7
  history.Append(Move("h9g7", true));

  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  zerod_plane = encoded_planes[15 * 8];
  EXPECT_EQ(zerod_plane.mask, 0ull);

  fifty_move_counter_plane = encoded_planes[15 * 8 + 1];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquares);
  EXPECT_EQ(fifty_move_counter_plane.value, 2.0f);
}

TEST(EncodePositionForNN, EncodeFormat) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/9/9/9/9/9/9/5K3 w - - 0 1");
  history.Reset(board, 0, 1);

  int transform;
  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, NoTransform);

  InputPlane our_king_plane = encoded_planes[6];
  EXPECT_EQ(our_king_plane.mask, 1ull << 5);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[13];
  EXPECT_EQ(their_king_plane.mask, __uint128_t(1) << 84);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeFormat2) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("5k3/9/9/9/9/9/9/9/9/3K5 w - - 0 1");
  history.Reset(board, 0, 1);

  // Their king offside, but not ours.
  int transform;
  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, NoTransform);

  InputPlane our_king_plane = encoded_planes[6];
  EXPECT_EQ(our_king_plane.mask, 1ull << 3);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[13];
  EXPECT_EQ(their_king_plane.mask, __uint128_t(1) << 86);
  EXPECT_EQ(their_king_plane.value, 1.0f);

  board.SetFromFen("3k5/9/9/9/9/9/9/9/9/5K3 w - - 0 1");
  history.Reset(board, 0, 1);

  // Our king offside, but theirs is not.
  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, FlipTransform);

  our_king_plane = encoded_planes[6];
  EXPECT_EQ(our_king_plane.mask, 1ull << 3);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  their_king_plane = encoded_planes[13];
  EXPECT_EQ(their_king_plane.mask, __uint128_t(1) << 86);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
