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

#include "chess/board.h"

#include <gtest/gtest.h>

#include <iostream>

#include "chess/bitboard.h"

#include "utils/exception.h"

namespace lczero {

TEST(BoardSquare, BoardSquare) {
  {
    auto x = BoardSquare(ChessBoard::C1);
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare("c1");
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    x.Mirror();
    EXPECT_EQ(x.row(), 8);
    EXPECT_EQ(x.col(), 2);
  }
}

TEST(ChessBoard, IllegalPawnPosition) {
  ChessBoard board;
  EXPECT_THROW(board.SetFromFen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P2PP1P1P/1C5C1/9/RNBAKABNR w");,
               Exception);
}

TEST(ChessBoard, PseudolegalMovesStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  board.Mirror();
  auto moves = board.GeneratePseudolegalMoves();

  EXPECT_EQ(moves.size(), 44);
}

TEST(ChessBoard, PartialFen) {
  ChessBoard board;
  int rule50ply;
  int gameply;
  board.SetFromFen("rnbakabnr//1c5c1/p1p1p1p1p///P1P1P1P1P/1C2K2C1", &rule50ply, &gameply);
  auto moves = board.GeneratePseudolegalMoves();

  EXPECT_EQ(moves.size(), 28);
  EXPECT_EQ(rule50ply, 0);
  EXPECT_EQ(gameply, 1);
}

TEST(ChessBoard, PartialFenWithSpaces) {
  ChessBoard board;
  int rule50ply;
  int gameply;
  board.SetFromFen("    rnbakabnr//1c5c1/p1p1p1p1p///P1P1P1P1P/1C2K2C1    w   ", &rule50ply, &gameply);
  auto moves = board.GeneratePseudolegalMoves();

  EXPECT_EQ(moves.size(), 28);
  EXPECT_EQ(rule50ply, 0);
  EXPECT_EQ(gameply, 1);
}

namespace {
int Perft(const ChessBoard& board, int max_depth, bool dump = false,
          int depth = 0) {
  if (depth == max_depth) return 1;
  int total_count = 0;
  auto moves = board.GeneratePseudolegalMoves();

  auto legal_moves = board.GenerateLegalMoves();
  auto iter = legal_moves.begin();

  for (const auto& move : moves) {
    auto new_board = board;
    new_board.ApplyMove(move);
    if (!new_board.IsLegalMove(move)) {
      if (iter != legal_moves.end()) {
        EXPECT_NE(iter->as_packed_int(), move.as_packed_int())
            << board.DebugString() << "legal:[" << iter->as_string()
            << "]==pseudo:(" << move.as_string() << ") Under check:\n"
            << new_board.DebugString();
      }
      continue;
    }

    EXPECT_EQ(iter->as_packed_int(), move.as_packed_int())
        << board.DebugString() << "legal:[" << iter->as_string() << "]pseudo:("
        << move.as_string() << ") after:\n"
        << new_board.DebugString();

    new_board.Mirror();
    ++iter;
    int count = Perft(new_board, max_depth, dump, depth + 1);
    if (dump && depth == 0) {
      Move m = move;
      if (board.flipped()) m.Mirror();
      std::cerr << m.as_string() << ": " << count << '\n';
    }
    total_count += count;
  }

  EXPECT_EQ(iter, legal_moves.end());
  return total_count;
}
}  // namespace

TEST(ChessBoard, MoveGenStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);

  EXPECT_EQ(Perft(board, 1), 44);
  EXPECT_EQ(Perft(board, 2), 1920);
  EXPECT_EQ(Perft(board, 3), 79666);
  EXPECT_EQ(Perft(board, 4), 3290240);
  EXPECT_EQ(Perft(board, 5), 133312995);
}

TEST(ChessBoard, MoveGenPosition2) {
  ChessBoard board;
  board.SetFromFen("r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w");

  EXPECT_EQ(Perft(board, 1), 38);
  EXPECT_EQ(Perft(board, 2), 1128);
  EXPECT_EQ(Perft(board, 3), 43929);
  EXPECT_EQ(Perft(board, 4), 1339047);
  EXPECT_EQ(Perft(board, 5), 53112976);
}

TEST(ChessBoard, MoveGenPosition3) {
  ChessBoard board;
  board.SetFromFen("1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w");

  EXPECT_EQ(Perft(board, 1), 7);
  EXPECT_EQ(Perft(board, 2), 281);
  EXPECT_EQ(Perft(board, 3), 8620);
  EXPECT_EQ(Perft(board, 4), 326201);
  EXPECT_EQ(Perft(board, 5), 10369923);
}

TEST(ChessBoard, MoveGenPosition4) {
  ChessBoard board;
  board.SetFromFen("5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w");

  EXPECT_EQ(Perft(board, 1), 25);
  EXPECT_EQ(Perft(board, 2), 424);
  EXPECT_EQ(Perft(board, 3), 9850);
  EXPECT_EQ(Perft(board, 4), 202884);
  EXPECT_EQ(Perft(board, 5), 4739553);
}

TEST(ChessBoard, MoveGenPosition5) {
  ChessBoard board;
  board.SetFromFen("CRN1k1b2/3ca4/4ba3/9/2nr5/9/9/4B4/4A4/4KA3 w");

  EXPECT_EQ(Perft(board, 1), 28);
  EXPECT_EQ(Perft(board, 2), 516);
  EXPECT_EQ(Perft(board, 3), 14808);
  EXPECT_EQ(Perft(board, 4), 395483);
  EXPECT_EQ(Perft(board, 5), 11842230);
}

TEST(ChessBoard, MoveGenPosition6) {
  ChessBoard board;
  board.SetFromFen("R1N1k1b2/9/3aba3/9/2nr5/2B6/9/4B4/4A4/4KA3 w");

  EXPECT_EQ(Perft(board, 1), 21);
  EXPECT_EQ(Perft(board, 2), 364);
  EXPECT_EQ(Perft(board, 3), 7626);
  EXPECT_EQ(Perft(board, 4), 162837);
  EXPECT_EQ(Perft(board, 5), 3500505);
}

TEST(ChessBoard, MoveGenPosition7) {
  ChessBoard board;
  board.SetFromFen("C1nNk4/9/9/9/9/9/n1pp5/B3C4/9/3A1K3 w");

  EXPECT_EQ(Perft(board, 1), 28);
  EXPECT_EQ(Perft(board, 2), 222);
  EXPECT_EQ(Perft(board, 3), 6241);
  EXPECT_EQ(Perft(board, 4), 64971);
  EXPECT_EQ(Perft(board, 5), 1914306);
}

TEST(ChessBoard, MoveGenPosition8) {
  ChessBoard board;
  board.SetFromFen("4ka3/4a4/9/9/4N4/p8/9/4C3c/7n1/2BK5 w");

  EXPECT_EQ(Perft(board, 1), 23);
  EXPECT_EQ(Perft(board, 2), 345);
  EXPECT_EQ(Perft(board, 3), 8124);
  EXPECT_EQ(Perft(board, 4), 149272);
  EXPECT_EQ(Perft(board, 5), 3513104);
}

TEST(ChessBoard, MoveGenPosition9) {
  ChessBoard board;
  board.SetFromFen("2b1ka3/9/b3N4/4n4/9/9/9/4C4/2p6/2BK5 w");

  EXPECT_EQ(Perft(board, 1), 21);
  EXPECT_EQ(Perft(board, 2), 195);
  EXPECT_EQ(Perft(board, 3), 3883);
  EXPECT_EQ(Perft(board, 4), 48060);
  EXPECT_EQ(Perft(board, 5), 933096);
}

TEST(ChessBoard, MoveGenPosition10) {
  ChessBoard board;
  board.SetFromFen("1C2ka3/9/C1Nab1n2/p3p3p/6p2/9/P3P3P/3AB4/3p2c2/c1BAK4 w");

  EXPECT_EQ(Perft(board, 1), 30);
  EXPECT_EQ(Perft(board, 2), 830);
  EXPECT_EQ(Perft(board, 3), 22787);
  EXPECT_EQ(Perft(board, 4), 649866);
  EXPECT_EQ(Perft(board, 5), 17920736);
}

TEST(ChessBoard, MoveGenPosition11) {
  ChessBoard board;
  board.SetFromFen("CnN1k1b2/c3a4/4ba3/9/2nr5/9/9/4C4/4A4/4KA3 w");

  EXPECT_EQ(Perft(board, 1), 19);
  EXPECT_EQ(Perft(board, 2), 583);
  EXPECT_EQ(Perft(board, 3), 11714);
  EXPECT_EQ(Perft(board, 4), 376467);
  EXPECT_EQ(Perft(board, 5), 8148177);
}

TEST(ChessBoard, HasMatingMaterialStartPosition) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialBareKings) {
  ChessBoard board;
  board.SetFromFen("3k5/9/9/9/9/9/9/9/9/5K3 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialAdvisorBishop) {
  ChessBoard board;
  board.SetFromFen("3k5/4a4/9/9/9/9/9/9/4A4/3A1K3 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("3k5/4a4/9/9/9/9/9/5A3/4A4/2B2K3 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialRookCannonKnight) {
  ChessBoard board;
  board.SetFromFen("3k5/4a4/9/9/9/9/9/5A3/R3A4/2B2K3 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("3k5/4a4/8c/9/9/9/9/5A3/4A4/2B2K3 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("3k5/4a4/9/9/9/9/9/N4A3/4A2N1/2B2K3 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

namespace {
void TestInvalid(std::string fen) {
  ChessBoard board;
  try {
    board.SetFromFen(fen);
    FAIL() << "Invalid Fen accepted: " + fen + "\n";
  } catch (...) {
    SUCCEED();
  }
}
}  // namespace


TEST(ChessBoard, InvalidFEN) {
  TestInvalid("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P2PP1P1P/1C5C1/9/RNBAKABNR w");
  TestInvalid("rrnbakabnr/9/1c5c1/p3p1p1p/3p5/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w");
  TestInvalid("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6A2/RNBAK1BNR w");
  TestInvalid("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6B2/RNBAKA1NR w");
  TestInvalid("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6K2/RNBA1ABNR w");
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
