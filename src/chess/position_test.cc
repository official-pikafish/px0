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

#include "chess/position.h"

#include <gtest/gtest.h>

#include <iostream>

#include "utils/string.h"

namespace lczero {

TEST(Position, SetFenGetFen) {
  std::vector<Position> positions;
  ChessBoard board;
  std::vector<std::string> source_fens = {
      "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
      "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 1 1",
      "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w - - 0 1",
      "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w - - 2 30",
      "CRN1k1b2/3ca4/4ba3/9/2nr5/9/9/4B4/4A4/4KA3 w - - 1 8",
      "R1N1k1b2/9/3aba3/9/2nr5/2B6/9/4B4/4A4/4KA3 w - - 0 10",
      "C1nNk4/9/9/9/9/9/n1pp5/B3C4/9/3A1K3 w - - 0 1",
      "4ka3/4a4/9/9/4N4/p8/9/4C3c/7n1/2BK5 w - - 0 1"};
  for (size_t i = 0; i < source_fens.size(); i++) {
    board.Clear();
    PositionHistory history;
    int no_capture_ply;
    int game_move;
    board.SetFromFen(source_fens[i], &no_capture_ply, &game_move);
    history.Reset(board, no_capture_ply,
                  2 * game_move - (board.flipped() ? 1 : 2));
    Position pos = history.Last();
    std::string target_fen = PositionToFen(pos);
    EXPECT_EQ(source_fens[i], target_fen);
  }
}

TEST(PositionHistory, ComputeLastMoveRepetitions1) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6c2/9/9/9/6R2/9/5K3 b");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 1);
}

TEST(PositionHistory, ComputeLastMoveRepetitions2) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6c2/9/9/9/6R2/9/5K3 b");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 2);
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveCurent) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBefore) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6rC1/9/9/9/5R3/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("f2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveOlder) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6rC1/9/9/9/5R3/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6b6"));
  history.Append(history.Last().GetBoard().ParseMove("f2b2"));
  history.Append(history.Last().GetBoard().ParseMove("b6h6"));
  history.Append(history.Last().GetBoard().ParseMove("b2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBeforeZero) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6f6"));
  history.Append(history.Last().GetBoard().ParseMove("g2f2"));
  history.Append(history.Last().GetBoard().ParseMove("f6g6"));
  history.Append(history.Last().GetBoard().ParseMove("f2g2"));
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveNeverRepeated) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6c6"));
  history.Append(history.Last().GetBoard().ParseMove("g2f2"));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, RuleJudgeWhiteChase) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g6h6"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("h6g6"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  EXPECT_EQ(history.RuleJudge(), GameResult::BLACK_WON);
}

TEST(PositionHistory, RuleJudgeBlackChase) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/7r1/9/9/9/9/6C2/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("h7g7"));
  history.Append(history.Last().GetBoard().ParseMove("g2h2"));
  history.Append(history.Last().GetBoard().ParseMove("g7h7"));
  history.Append(history.Last().GetBoard().ParseMove("h2g2"));
  EXPECT_EQ(history.RuleJudge(), GameResult::WHITE_WON);

  board.SetFromFen("1rbakabnr/9/2n6/p1p3p1p/c8/4C4/P1P1P1PcP/1C2B1N2/3N5/R2AKABR1 w");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("a0c0"));
  history.Append(history.Last().GetBoard().ParseMove("a5c5"));
  history.Append(history.Last().GetBoard().ParseMove("c0a0"));
  history.Append(history.Last().GetBoard().ParseMove("c5a5"));
  EXPECT_EQ(history.RuleJudge(), GameResult::BLACK_WON);
}

TEST(PositionHistory, RuleJudgeWhiteCheck) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("d9e9"));
  history.Append(history.Last().GetBoard().ParseMove("d2e2"));
  history.Append(history.Last().GetBoard().ParseMove("e9d9"));
  history.Append(history.Last().GetBoard().ParseMove("e2d2"));
  EXPECT_EQ(history.RuleJudge(), GameResult::BLACK_WON);
}

TEST(PositionHistory, RuleJudgeBlackCheck) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/4r4/9/9/9/9/9/9/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("e7f7"));
  history.Append(history.Last().GetBoard().ParseMove("f0e0"));
  history.Append(history.Last().GetBoard().ParseMove("f7e7"));
  history.Append(history.Last().GetBoard().ParseMove("e0f0"));
  EXPECT_EQ(history.RuleJudge(), GameResult::WHITE_WON);
}

TEST(PositionHistory, RuleJudgeDraw) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3k5/9/6r2/9/9/9/9/9/6R2/5K3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("g7h7"));
  history.Append(history.Last().GetBoard().ParseMove("g1h1"));
  history.Append(history.Last().GetBoard().ParseMove("h7g7"));
  history.Append(history.Last().GetBoard().ParseMove("h1g1"));
  EXPECT_EQ(history.RuleJudge(), GameResult::DRAW);

  board.SetFromFen("4c4/3k5/4b3b/9/9/2B4N1/4p4/3A5/2p1A4/5K3 w - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("h4g2"));
  history.Append(history.Last().GetBoard().ParseMove("e3f3"));
  history.Append(history.Last().GetBoard().ParseMove("g2h4"));
  history.Append(history.Last().GetBoard().ParseMove("f3e3"));
  EXPECT_EQ(history.RuleJudge(), GameResult::DRAW);

  board.SetFromFen("3k5/9/9/9/9/9/9/9/1r2ARn2/4K4 b");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("b1b0"));
  history.Append(history.Last().GetBoard().ParseMove("e1d0"));
  history.Append(history.Last().GetBoard().ParseMove("b0b1"));
  history.Append(history.Last().GetBoard().ParseMove("d0e1"));
  EXPECT_EQ(history.RuleJudge(), GameResult::DRAW);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
