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

#include "selfplay/loop.h"

#include <optional>

#include "neural/decoder.h"
#include "selfplay/tournament.h"
#include "trainingdata/reader.h"
#include "utils/configfile.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};

const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kPolicySubsDirId{"policy-substitutions", "",
                                "Directory with gzipped files are to use to "
                                "replace policy for some of the data."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with.", 't'};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kNewInputFormatId{
    "new-input-format", "",
    "Input format to convert training data to during rescoring."};
const OptionId kDeblunder{
    "deblunder", "",
    "If true, whether to use move Q information to infer a different Z value "
    "if the the selected move appears to be a blunder."};
const OptionId kDeblunderQBlunderThreshold{
    "deblunder-q-blunder-threshold", "",
    "The amount Q of played move needs to be worse than best move in order to "
    "assume the played move is a blunder."};
const OptionId kDeblunderQBlunderWidth{
    "deblunder-q-blunder-width", "",
    "Width of the transition between accepted temp moves and blunders."};
const OptionId kNnuePlainFileId{"nnue-plain-file", "",
                                "Append SF plain format training data to this "
                                "file. Will be generated if not there."};
const OptionId kNnueBestScoreId{"nnue-best-score", "",
                                "For the SF training data use the score of the "
                                "best move instead of the played one."};
const OptionId kNnueBestMoveId{
    "nnue-best-move", "",
    "For the SF training data record the best move instead of the played one. "
    "If set to true the generated files do not compress well."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};

const OptionId kLogFileId{"logfile", "LogFile",
  "Write log to that file. Special value <stderr> to "
  "output the log to the console."};

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 2062; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[2062];
  PolicySubNode* children[2062];
};

struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];
std::atomic<int> policy_dtm_bump(0);
std::map<uint64_t, PolicySubNode> policy_subs;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(const std::vector<V6TrainingData>& fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (int i = 0; i < fileContents.size(); i++) {
    auto& data = fileContents[i];
    DataAssert(
        data.input_format ==
            pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
        data.input_format ==
            pblczero::NetworkFormat::
                INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);
    DataAssert(data.best_d >= 0.0f && data.best_d <= 1.0f);
    DataAssert(data.root_d >= 0.0f && data.root_d <= 1.0f);
    DataAssert(data.best_q >= -1.0f && data.best_q <= 1.0f);
    DataAssert(data.root_q >= -1.0f && data.root_q <= 1.0f);
    DataAssert(data.root_m >= 0.0f);
    DataAssert(data.best_m >= 0.0f);
    DataAssert(data.plies_left >= 0.0f);
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            data.input_format))) {
      // At most one en-passant bit.
      DataAssert((data.side_to_move &
                  (data.side_to_move - 1)) == 0);
    } else {
      DataAssert(data.side_to_move >= 0 &&
                 data.side_to_move <= 1);
    }
    DataAssert(data.result_q >= -1 && data.result_q <= 1);
    DataAssert(data.result_d >= 0 && data.result_q <= 1);
    DataAssert(data.rule50_count >= 0 && data.rule50_count <= 120);
    float sum = 0.0f;
    for (int j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert(prob >= 0.0f && prob <= 1.0f || prob == -1.0f ||
                 std::isnan(prob));
      if (prob >= 0.0f) {
        sum += prob;
      }
      // Only check best_idx/played_idx for real v6 data.
      if (data.visits > 0) {
        // Best_idx and played_idx must be marked legal in probabilities.
        if (j == data.best_idx || j == data.played_idx) {
          DataAssert(prob >= 0.0f);
        }
      }
    }
    if (sum < 0.99f || sum > 1.01f) {
      throw Exception("Probability sum error is huge!");
    }
    DataAssert(data.best_idx <= 2062);
    DataAssert(data.played_idx <= 2062);
    DataAssert(data.played_q >= -1.0f && data.played_q <= 1.0f);
    DataAssert(data.played_d >= 0.0f && data.played_d <= 1.0f);
    DataAssert(data.played_m >= 0.0f);
    DataAssert(std::isnan(data.orig_q) ||
               data.orig_q >= -1.0f && data.orig_q <= 1.0f);
    DataAssert(std::isnan(data.orig_d) ||
               data.orig_d >= 0.0f && data.orig_d <= 1.0f);
    DataAssert(std::isnan(data.orig_m) || data.orig_m >= 0.0f);
    // TODO: if visits > 0 - assert best_idx/played_idx are valid in
    // probabilities.
  }
}

void Validate(const std::vector<V6TrainingData>& fileContents,
              const MoveList& moves) {
  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;
  auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
      fileContents[0].input_format);
  PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  for (int i = 0; i < moves.size(); i++) {
    int transform = TransformForPosition(input_format, history);
    // If real v6 data, can confirm that played_idx matches the inferred move.
    if (fileContents[i].visits > 0) {
      if (fileContents[i].played_idx != moves[i].as_nn_index(transform)) {
        throw Exception("Move performed is not listed as played.");
      }
    }
    // Move shouldn't be marked illegal unless there is 0 visits, which should
    // only happen if invariance_info is marked with the placeholder bit.
    if (!(fileContents[i].probabilities[moves[i].as_nn_index(transform)] >=
          0.0f) &&
        (fileContents[i].invariance_info & 64) == 0) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is an illegal move.");
    }
    history.Append(moves[i]);
  }
}

void ChangeInputFormat(int newInputFormat, V6TrainingData* data,
                       const PositionHistory& history) {
  data->input_format = newInputFormat;
  auto input_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(input_format, history, 8,
                                           FillEmptyHistory::NO, &transform);
  int plane_idx = 0;
  for (auto& plane : data->planes) {
    plane = FlipBoard(planes[plane_idx++].mask);
  }

  if ((data->invariance_info & 7) != transform) {
    // Probabilities need reshuffling.
    float newProbs[2062];
    std::fill(std::begin(newProbs), std::end(newProbs), -1);
    bool played_fixed = false;
    bool best_fixed = false;
    for (auto move : history.Last().GetBoard().GenerateLegalMoves()) {
      int i = move.as_nn_index(transform);
      int j = move.as_nn_index(data->invariance_info & 7);
      newProbs[i] = data->probabilities[j];
      // For V6 data only, the played/best idx need updating.
      if (data->visits > 0) {
        if (data->played_idx == j && !played_fixed) {
          data->played_idx = i;
          played_fixed = true;
        }
        if (data->best_idx == j && !best_fixed) {
          data->best_idx = i;
          best_fixed = true;
        }
      }
    }
    for (int i = 0; i < 2062; i++) {
      data->probabilities[i] = newProbs[i];
    }
  }

  const auto& position = history.Last();

  // Save the bits that aren't connected to the input_format.
  uint8_t invariance_mask = data->invariance_info & 0x78;
  // Other params.
  if (IsCanonicalFormat(input_format)) {
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    data->invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    data->side_to_move = position.IsBlackToMove() ? 1 : 0;
    data->invariance_info = 0;
  }
  // Put the mask back.
  data->invariance_info |= invariance_mask;
}

int ResultForData(const V6TrainingData& data) {
  // Ensure we aren't reprocessing some data that has had custom adjustments to
  // result training target applied.
  DataAssert(data.result_q == -1.0f || data.result_q == 1.0f ||
             data.result_q == 0.0f);
  // Paranoia - ensure int cast never breaks the value.
  DataAssert(data.result_q ==
             static_cast<float>(static_cast<int>(data.result_q)));
  return static_cast<int>(data.result_q);
}

std::string AsNnueString(const Position& p, Move best, Move played, float q,
                         int result, const ProcessFileFlags &flags) {
  // Filter out in check and pv captures.
  bool filtered = p.GetWhiteBoard().IsUnderCheck() ||
                  p.GetWhiteBoard().theirs().get(best.to());
  std::ostringstream out;
  out << "fen " << GetFen(p) << std::endl;
  if (p.IsBlackToMove()) best.Mirror(), played.Mirror();
  out << "move " << (flags.nnue_best_move ? best.as_string() : played.as_string()) << std::endl;
  // Formula from PR1477 adjuster for SF PawnValueEg.
  out << "score "
      << (filtered ? 32000 : round(660.6 * q / (1 - 0.9751875 * std::pow(q, 10))))
      << std::endl;
  out << "ply " << p.GetGamePly() << std::endl;
  out << "result " << result << std::endl;
  out << "e" << std::endl;
  return out.str();
}

void ProcessFile(const std::string& file, std::string outputDir, float distTemp, float distOffset,
                 int newInputFormat, std::string nnue_plain_file, ProcessFileFlags flags) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      TrainingDataReader reader(file);
      std::vector<V6TrainingData> fileContents;
      V6TrainingData data;
      while (reader.ReadChunk(&data)) {
        fileContents.push_back(data);
      }
      Validate(fileContents);
      MoveList moves;
      for (int i = 1; i < fileContents.size(); i++) {
        moves.push_back(
            DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                                PlanesFromTrainingData(fileContents[i - 1])));
        // All moves decoded are from the point of view of the side after the
        // move so need to mirror them all to be applicable to apply to the
        // position before.
        moves.back().Mirror();
      }
      Validate(fileContents, moves);
      games += 1;
      positions += fileContents.size();
      PositionHistory history;
      int rule50ply;
      int gameply;
      ChessBoard board;
      auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
          fileContents[0].input_format);
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      uint64_t rootHash = HashCat(board.Hash(), rule50ply);
      if (policy_subs.find(rootHash) != policy_subs.end()) {
        PolicySubNode* rootNode = &policy_subs[rootHash];
        for (int i = 0; i < fileContents.size(); i++) {
          if (rootNode->active) {
            /* Some logic for choosing a softmax to apply to better align the
            new policy with the old policy...
            double bestkld =
              std::numeric_limits<double>::max(); float besttemp = 1.0f;
            // Minima is usually in this range for 'better' data.
            for (float temp = 1.0f; temp < 3.0f; temp += 0.1f) {
              float soft[2062];
              float sum = 0.0f;
              for (int j = 0; j < 2062; j++) {
                if (rootNode->policy[j] >= 0.0) {
                  soft[j] = std::pow(rootNode->policy[j], 1.0f / temp);
                  sum += soft[j];
                } else {
                  soft[j] = -1.0f;
                }
              }
              double kld = 0.0;
              for (int j = 0; j < 2062; j++) {
                if (soft[j] >= 0.0) soft[j] /= sum;
                if (rootNode->policy[j] > 0.0 &&
                    fileContents[i].probabilities[j] > 0) {
                  kld += -1.0f * soft[j] *
                    std::log(fileContents[i].probabilities[j] / soft[j]);
                }
              }
              if (kld < bestkld) {
                bestkld = kld;
                besttemp = temp;
              }
            }
            std::cerr << i << " " << besttemp << " " << bestkld << std::endl;
            */
            for (int j = 0; j < 2062; j++) {
              /*
              if (rootNode->policy[j] >= 0.0) {
                std::cerr << i << " " << j << " " << rootNode->policy[j] << " "
                          << fileContents[i].probabilities[j] << std::endl;
              }
              */
              fileContents[i].probabilities[j] = rootNode->policy[j];
            }
          }
          if (i < fileContents.size() - 1) {
            int transform = TransformForPosition(input_format, history);
            int idx = moves[i].as_nn_index(transform);
            if (rootNode->children[idx] == nullptr) {
              break;
            }
            rootNode = rootNode->children[idx];
            history.Append(moves[i]);
          }
        }
      }

      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      int last_rescore = -1;
      orig_counts[ResultForData(fileContents[0]) + 1]++;
      fixed_counts[ResultForData(fileContents[0]) + 1]++;
      for (int i = 0; i < moves.size(); i++) {
        history.Append(moves[i]);
      }

      if (distTemp != 1.0f || distOffset != 0.0f) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        int move_index = 0;
        for (auto& chunk : fileContents) {
          const auto& board = history.Last().GetBoard();
          std::vector<bool> boost_probs(2062, false);
          int boost_count = 0;

          float sum = 0.0;
          int prob_index = 0;
          float preboost_sum = 0.0f;
          for (auto& prob : chunk.probabilities) {
            float offset = distOffset;
            prob_index++;
            if (prob < 0 || std::isnan(prob)) continue;
            prob = std::max(0.0f, prob + offset);
            prob = std::pow(prob, 1.0f / distTemp);
            sum += prob;
          }
          prob_index = 0;
          float boost_sum = 0.0f;
          for (auto& prob : chunk.probabilities) {
            prob_index++;
            if (prob < 0 || std::isnan(prob)) continue;
            prob /= sum;
          }
          if (boost_count > 0) {
            policy_nobump_total_hist[(int)(preboost_sum * 10)]++;
            policy_bump_total_hist[(int)(boost_sum * 10)]++;
          }
          history.Append(moves[move_index]);
          move_index++;
        }
      }

      // Make move_count field plies_left for moves left head.
      int offset = 0;
      bool all_draws = true;
      for (auto& chunk : fileContents) {
        // plies_left can't be 0 for real v5 data, so if it is 0 it must be a v4
        // conversion, and we should populate it ourselves with a better
        // starting estimate.
        if (chunk.plies_left == 0.0f) {
          chunk.plies_left = (int)(fileContents.size() - offset);
        }
        offset++;
        all_draws = all_draws && (ResultForData(chunk) == 0);
      }

      // Deblunder only works from v6 data onwards. We therefore check
      // the visits field which is 0 if we're dealing with upgraded data.
      if (deblunderEnabled && fileContents.back().visits > 0) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (int i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
        }
        float activeZ[3] = {fileContents.back().result_q,
                            fileContents.back().result_d,
                            fileContents.back().plies_left};
        bool deblunderingStarted = false;
        while (true) {
          auto& cur = fileContents[history.GetLength() - 1];
          // A blunder is defined by the played move being worse than the
          // best move by a defined threshold, missing a forced win, or
          // playing into a proven loss without being forced.
          bool deblunderTriggerThreshold =
              (cur.best_q - cur.played_q >
               deblunderQBlunderThreshold - deblunderQBlunderWidth / 2.0);
          bool deblunderTriggerTerminal =
              (cur.best_q > -1 && cur.played_q < 1 &&
               ((cur.best_q == 1 && ((cur.invariance_info & 8) != 0)) ||
                cur.played_q == -1));
          if (deblunderTriggerThreshold || deblunderTriggerTerminal) {
            float newZRatio = 1.0f;
            // If width > 0 and the deblunder didn't involve a terminal
            // position, we apply a soft threshold by averaging old and new Z.
            if (deblunderQBlunderWidth > 0 && !deblunderTriggerTerminal) {
              newZRatio = std::min(1.0f, (cur.best_q - cur.played_q -
                                          deblunderQBlunderThreshold) /
                                                 deblunderQBlunderWidth +
                                             0.5f);
            }
            // Instead of averaging, a randomization can be applied here with
            // newZRatio = newZRatio > rand( [0, 1) ) ? 1.0f : 0.0f;
            activeZ[0] = (1 - newZRatio) * activeZ[0] + newZRatio * cur.best_q;
            activeZ[1] = (1 - newZRatio) * activeZ[1] + newZRatio * cur.best_d;
            activeZ[2] = (1 - newZRatio) * activeZ[2] + newZRatio * cur.best_m;
            deblunderingStarted = true;
            blunders += 1;
            /* std::cout << "Blunder detected. Best move q=" << cur.best_q <<
             " played move q=" << cur.played_q; */
          }
          if (deblunderingStarted) {
            /*
            std::cerr << "Deblundering: "
                      << fileContents[history.GetLength() - 1].best_q << " "
                      << fileContents[history.GetLength() - 1].best_d << " "
                      << (int)fileContents[history.GetLength() - 1].result << "
            "
                      << (int)activeZ << std::endl;
                      */
            fileContents[history.GetLength() - 1].result_q = activeZ[0];
            fileContents[history.GetLength() - 1].result_d = activeZ[1];
            fileContents[history.GetLength() - 1].plies_left = activeZ[2];
          }
          if (history.GetLength() == 1) break;
          // Q values are always from the player to move.
          activeZ[0] = -activeZ[0];
          // Estimated remaining plies left has to be increased.
          activeZ[2] += 1.0f;
          history.Pop();
        }
      }
      if (newInputFormat != -1) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        ChangeInputFormat(newInputFormat, &fileContents[0], history);
        for (int i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          ChangeInputFormat(newInputFormat, &fileContents[i + 1], history);
        }
      }

      if (!outputDir.empty()) {
        std::string fileName = file.substr(file.find_last_of("/\\") + 1);
        TrainingDataWriter writer(outputDir + "/" + fileName);
        for (auto chunk : fileContents) {
          // Don't save chunks that just provide move history.
          if ((chunk.invariance_info & 64) == 0) {
            writer.WriteChunk(chunk);
          }
        }
      }

      // Output data in Stockfish plain format.
      if (!nnue_plain_file.empty()) {
        static Mutex mutex;
        std::ostringstream out;
        pblczero::NetworkFormat::InputFormat format;
        if (newInputFormat != -1) {
          format =
              static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);
        } else {
          format = input_format;
        }
        PopulateBoard(format, PlanesFromTrainingData(fileContents[0]), &board,
                      &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (int i = 0; i < fileContents.size(); i++) {
          auto chunk = fileContents[i];
          Position p = history.Last();
          if (chunk.visits > 0) {
            // Format is v6 and position is evaluated.
            Move best = MoveFromNNIndex(chunk.best_idx, TransformForPosition(format, history));
            Move played = MoveFromNNIndex(chunk.played_idx, TransformForPosition(format, history));
            float q = flags.nnue_best_score ? chunk.best_q : chunk.played_q;
            out << AsNnueString(p, best, played, q, round(chunk.result_q), flags);
          } else if (i < moves.size()) {
            out << AsNnueString(p, moves[i], moves[i], chunk.best_q,
                                round(chunk.result_q), flags);
          }
          if (i < moves.size()) {
            history.Append(moves[i]);
          }
        }
        std::ofstream file;
        Mutex::Lock lock(mutex);
        file.open(nnue_plain_file, std::ios_base::app);
        if (file.is_open()) {
          file << out.str();
          file.close();
        }
      }
    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
      if (flags.delete_files) {
        std::cerr << "It will be deleted." << std::endl;
      }
    }
  }
  if (flags.delete_files) {
    remove(file.c_str());
  }
}

void ProcessFiles(const std::vector<std::string>& files, std::string outputDir,
                  float distTemp, float distOffset,
                  int newInputFormat, int offset, int mod,
                  std::string nnue_plain_file, ProcessFileFlags flags) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (int i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], outputDir, distTemp, distOffset,
                newInputFormat, nnue_plain_file, flags);
  }
}

void BuildSubs(const std::vector<std::string>& files) {
  for (auto& file : files) {
    TrainingDataReader reader(file);
    std::vector<V6TrainingData> fileContents;
    V6TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    Validate(fileContents);
    MoveList moves;
    for (int i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                              PlanesFromTrainingData(fileContents[i - 1])));
      // All moves decoded are from the point of view of the side after the
      // move so need to mirror them all to be applicable to apply to the
      // position before.
      moves.back().Mirror();
    }
    Validate(fileContents, moves);

    // Subs are 'valid'.
    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
        fileContents[0].input_format);
    PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                  &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    uint64_t rootHash = HashCat(board.Hash(), rule50ply);
    PolicySubNode* rootNode = &policy_subs[rootHash];
    for (int i = 0; i < fileContents.size(); i++) {
      if ((fileContents[i].invariance_info & 64) == 0) {
        rootNode->active = true;
        for (int j = 0; j < 2062; j++) {
          rootNode->policy[j] = fileContents[i].probabilities[j];
        }
      }
      if (i < fileContents.size() - 1) {
        int transform = TransformForPosition(input_format, history);
        int idx = moves[i].as_nn_index(transform);
        if (rootNode->children[idx] == nullptr) {
          rootNode->children[idx] = new PolicySubNode();
        }
        rootNode = rootNode->children[idx];
        history.Append(moves[i]);
      }
    }
  }
}

}  // namespace

ConvertLoop::ConvertLoop() {}

ConvertLoop::~ConvertLoop() {}

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void ConvertLoop::RunLoop() {
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  for (int i = 0; i < 11; i++) policy_bump_total_hist[i] = 0;
  for (int i = 0; i < 11; i++) policy_nobump_total_hist[i] = 0;
  options_.Add<StringOption>(kInputDirId);
  options_.Add<StringOption>(kOutputDirId);
  options_.Add<StringOption>(kPolicySubsDirId);
  options_.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options_.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options_.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options_.Add<IntOption>(kNewInputFormatId, -1, 256) = -1;
  options_.Add<BoolOption>(kDeblunder) = false;
  options_.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options_.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options_.Add<StringOption>(kNnuePlainFileId);
  options_.Add<BoolOption>(kNnueBestScoreId) = true;
  options_.Add<BoolOption>(kNnueBestMoveId) = false;
  options_.Add<BoolOption>(kDeleteFilesId) = true;

  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;

  if (options_.GetOptionsDict().IsDefault<std::string>(kOutputDirId) &&
      options_.GetOptionsDict().IsDefault<std::string>(kNnuePlainFileId)) {
    std::cerr << "Must provide an output dir or NNUE plain file." << std::endl;
    return;
  }

  deblunderEnabled = options_.GetOptionsDict().Get<bool>(kDeblunder);
  deblunderQBlunderThreshold =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderThreshold);
  deblunderQBlunderWidth =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderWidth);

  auto policySubsDir =
      options_.GetOptionsDict().Get<std::string>(kPolicySubsDirId);
  if (policySubsDir.size() != 0) {
    auto policySubFiles = GetFileList(policySubsDir);
    for (int i = 0; i < policySubFiles.size(); i++) {
      policySubFiles[i] = policySubsDir + "/" + policySubFiles[i];
    }
    BuildSubs(policySubFiles);
  }

  auto inputDir = options_.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.size() == 0) {
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (int i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  int threads = options_.GetOptionsDict().Get<int>(kThreadsId);
  ProcessFileFlags flags;
  flags.delete_files = options_.GetOptionsDict().Get<bool>(kDeleteFilesId);
  flags.nnue_best_score = options_.GetOptionsDict().Get<bool>(kNnueBestScoreId);
  flags.nnue_best_move = options_.GetOptionsDict().Get<bool>(kNnueBestMoveId);
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([this, offset_val, files, threads, flags]() {
        ProcessFiles(
            files,
            options_.GetOptionsDict().Get<std::string>(kOutputDirId),
            options_.GetOptionsDict().Get<float>(kTempId),
            options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
            options_.GetOptionsDict().Get<int>(kNewInputFormatId),
            offset_val, threads,
            options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
            flags);
      });
    }
    for (int i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(
        files,
        options_.GetOptionsDict().Get<std::string>(kOutputDirId),
        options_.GetOptionsDict().Get<float>(kTempId),
        options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
        options_.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
        options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId), flags);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Rescores performed: " << rescored << std::endl;
  std::cout << "Cumulative outcome change: " << delta << std::endl;
  std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  std::cout << "Secondary rescores performed used dtz: " << rescored3
            << std::endl;
  std::cout << "Blunders picked up by deblunder threshold: " << blunders
            << std::endl;
  std::cout << "Number of policy values boosted by dtz or dtm " << policy_bump
            << std::endl;
  std::cout << "Number of policy values boosted by dtm " << policy_dtm_bump
            << std::endl;
  std::cout << "Orig policy_sum dist of boost candidate:";
  std::cout << std::endl;
  int event_sum = 0;
  for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i];
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_nobump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Boosted policy_sum dist of boost candidate:";
  std::cout << std::endl;
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_bump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
            << " W: " << orig_counts[2] << std::endl;
  std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
            << " W: " << fixed_counts[2] << std::endl;
}

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<BoolOption>(kInteractiveId) = false;
  options_.Add<StringOption>(kLogFileId);

  if (!options_.ProcessAllFlags()) return;
  
  Logging::Get().SetFilename(options_.GetOptionsDict().Get<std::string>(kLogFileId));

  if (options_.GetOptionsDict().Get<bool>(kInteractiveId)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetUciOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];

  // Initialize variables.
  float percentage = -1;
  std::optional<float> elo;
  std::optional<float> los;

  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}

}  // namespace lczero
