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
#include "utils/bititer.h"
#include "utils/logging.h"

namespace lczero {
namespace {

constexpr std::array<float, 2062> kLogPolicy { };

constexpr std::array<float, 90> kRooks = { };
constexpr std::array<float, 90> kAdvisors = { };
constexpr std::array<float, 90> kCannons = { };
constexpr std::array<float, 90> kPawns = { };
constexpr std::array<float, 90> kKnights = { };
constexpr std::array<float, 90> kBishops = { };
constexpr std::array<float, 90> kKings = { };

float DotProduct(__uint128_t plane, const std::array<float, 90>& weights) {
  float result = 0.0f;
  for (auto idx : IterateBits(plane)) result += weights[idx];
  return result;
}

class TrivialNetworkComputation : public NetworkComputation {
 public:
  void AddInput(InputPlanes&& input) override {
    float q = 0.0f;
    q += DotProduct(input[0].mask, kRooks);
    q -= DotProduct(MirrorBoard(input[7].mask), kRooks);
    q += DotProduct(input[1].mask, kAdvisors);
    q -= DotProduct(MirrorBoard(input[8].mask), kAdvisors);
    q += DotProduct(input[2].mask, kCannons);
    q -= DotProduct(MirrorBoard(input[9].mask), kCannons);
    q += DotProduct(input[3].mask, kPawns);
    q -= DotProduct(MirrorBoard(input[10].mask), kPawns);
    q += DotProduct(input[4].mask, kKnights);
    q -= DotProduct(MirrorBoard(input[11].mask), kKnights);
    q += DotProduct(input[5].mask, kBishops);
    q -= DotProduct(MirrorBoard(input[12].mask), kBishops);
    q += DotProduct(input[6].mask, kKings);
    q -= DotProduct(MirrorBoard(input[13].mask), kKings);
    // Multiply Q by 10, otherwise evals too low. :-/
    q_.push_back(2.0f / (1.0f + std::exp(q * -10.0f)) - 1.0f);
  }

  void ComputeBlocking() override {}

  int GetBatchSize() const override { return q_.size(); }

  float GetQVal(int sample) const override { return q_[sample]; }

  float GetDVal(int) const override { return 0.0f; }

  float GetMVal(int /* sample */) const override { return 0.0f; }

  float GetPVal(int /* sample */, int move_id) const override {
    return kLogPolicy[move_id];
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
      pblczero::NetworkFormat::MOVES_LEFT_NONE};
};
}  // namespace

std::unique_ptr<Network> MakeTrivialNetwork(
    const std::optional<WeightsFile>& /*weights*/, const OptionsDict& options) {
  return std::make_unique<TrivialNetwork>(options);
}

REGISTER_NETWORK("trivial", MakeTrivialNetwork, 4)

}  // namespace lczero
