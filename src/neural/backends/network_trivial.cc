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

// The trivial backend: a very fast CPU-only evaluator.
// Value: material balance of the side to move.
// Policy: uniform over the legal moves.

#include <cmath>
#include <memory>

#include "chess/board.h"
#include "neural/backend.h"
#include "neural/register.h"

namespace lczero {
namespace {

float MaterialScore(const ChessBoard& board, BitBoard pieces, float score) {
  return ((board.ours() & pieces).count() -
          (board.theirs() & pieces).count()) *
         score;
}

float Evaluate(const ChessBoard& board) {
  const float q = MaterialScore(board, board.rooks(), 0.18181818181818182f) +
                  MaterialScore(board, board.advisors(), 0.03636363636363636f) +
                  MaterialScore(board, board.cannons(), 0.10090909090909091f) +
                  MaterialScore(board, board.pawns(), 0.01818181818181818f) +
                  MaterialScore(board, board.knights(), 0.08090909090909090f) +
                  MaterialScore(board, board.bishops(), 0.05454545454545454f);
  // Multiply Q by 10, otherwise evals too low. :-/
  return 2.0f / (1.0f + std::exp(q * -10.0f)) - 1.0f;
}

class TrivialComputation : public BackendComputation {
 public:
  size_t UsedBatchSize() const override { return batch_size_; }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    // The evaluation is so cheap that it is done right away; the legal moves
    // span is not guaranteed to outlive this call anyway.
    if (result.q) *result.q = Evaluate(pos.pos.back().GetBoard());
    if (result.d) *result.d = 0.0f;
    if (result.m) *result.m = 0.0f;
    if (!result.p.empty()) {
      const float prob = 1.0f / result.p.size();
      for (float& p : result.p) p = prob;
    }
    ++batch_size_;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {}

 private:
  size_t batch_size_ = 0;
};

class TrivialBackend : public Backend {
 public:
  explicit TrivialBackend(const OptionsDict& options) {
    UpdateConfiguration(options);
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = false,
                             .runs_on_cpu = false,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 256,
                             .maximum_batch_size = 1024};
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<TrivialComputation>();
  }
};

class TrivialBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return 4; }
  std::string_view GetName() const override { return "trivial"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<TrivialBackend>(options);
  }
};

REGISTER_BACKEND(TrivialBackendFactory)

}  // namespace
}  // namespace lczero
