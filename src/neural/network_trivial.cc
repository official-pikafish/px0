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

// This is the trivial backend, which.
// Uses idea from
// https://www.chessprogramming.org/Simplified_Evaluation_Function
// for Q (but coefficients are "trained" from 1000 arbitrary test60 games).
// Returns the same P vector always ("trained" from 1 hour of test60 games).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>

#include "neural/factory.h"
#include "chess/bitboard.h"

namespace lczero {
namespace {

static inline float MaterialScore(const __uint128_t& maskOurs, const __uint128_t& maskTheirs, double score) {
  return (BitBoard(maskOurs).count() - BitBoard(maskTheirs).count()) * score;
}

class TrivialNetworkComputation : public NetworkComputation {
 public:
  void AddInput(InputPlanes&& input) override {
    float q =
        // Rook
        MaterialScore(input[0].mask, input[7].mask, 0.18181818181818182) +
        // Advisor
        MaterialScore(input[1].mask, input[8].mask, 0.03636363636363636) +
        // Cannon
        MaterialScore(input[2].mask, input[9].mask, 0.10090909090909091) +
        // Pawn
        MaterialScore(input[3].mask, input[10].mask, 0.01818181818181818) +
        // Knight
        MaterialScore(input[4].mask, input[11].mask, 0.08090909090909090) +
        // Bishop
        MaterialScore(input[5].mask, input[12].mask, 0.05454545454545454);
    // Multiply Q by 10, otherwise evals too low. :-/
    q_.push_back(2.0f / (1.0f + std::exp(q * -10.0f)) - 1.0f);
  }

  void ComputeBlocking() override {}

  int GetBatchSize() const override { return q_.size(); }

  float GetQVal(int sample) const override { return q_[sample]; }

  float GetDVal(int) const override { return 0.0f; }

  float GetMVal(int /* sample */) const override { return 0.0f; }

  float GetPVal(int /* sample */, [[maybe_unused]]int move_id) const override {
    return 0.0f;
  }

 private:
  std::vector<float> q_;
};

class TrivialNetwork : public Network {
 public:
  TrivialNetwork(const OptionsDict& options)
      : capabilities_{
            static_cast<pblczero::NetworkFormat::InputFormat>(
                options.GetOrDefault<int>(
                    "input_mode",
                    pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE)),
            pblczero::NetworkFormat::OUTPUT_CLASSICAL,
            pblczero::NetworkFormat::MOVES_LEFT_NONE} {}
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<TrivialNetworkComputation>();
  }
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  NetworkCapabilities capabilities_{
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
      pblczero::NetworkFormat::OUTPUT_CLASSICAL,
      pblczero::NetworkFormat::MOVES_LEFT_NONE};
};
}  // namespace

std::unique_ptr<Network> MakeTrivialNetwork(
    const std::optional<WeightsFile>& /*weights*/, const OptionsDict& options) {
  return std::make_unique<TrivialNetwork>(options);
}

REGISTER_NETWORK("trivial", MakeTrivialNetwork, 4)

}  // namespace lczero
