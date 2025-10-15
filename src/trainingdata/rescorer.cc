/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "trainingdata/rescorer.h"

#include <algorithm>
#include <optional>
#include <span>
#include <sstream>

#include "neural/decoder.h"
#include "trainingdata/reader.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
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

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 2062; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[2062];
  PolicySubNode* children[2062];
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::map<uint64_t, PolicySubNode> policy_subs;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(std::span<const V6TrainingData> fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (size_t i = 0; i < fileContents.size(); i++) {
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
      DataAssert((data.side_to_move & (data.side_to_move - 1)) == 0);
    } else {
      DataAssert(data.side_to_move <= 1);
    }
    DataAssert(data.result_q >= -1 && data.result_q <= 1);
    DataAssert(data.result_d >= 0 && data.result_q <= 1);
    DataAssert(data.rule50_count <= 120);
    float sum = 0.0f;
    for (size_t j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert((prob >= 0.0f && prob <= 1.0f) || prob == -1.0f ||
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
               (data.orig_q >= -1.0f && data.orig_q <= 1.0f));
    DataAssert(std::isnan(data.orig_d) ||
               (data.orig_d >= 0.0f && data.orig_d <= 1.0f));
    DataAssert(std::isnan(data.orig_m) || data.orig_m >= 0.0f);
    // TODO: if visits > 0 - assert best_idx/played_idx are valid in
    // probabilities.
  }
}

void Validate(std::span<const V6TrainingData> fileContents,
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
  for (size_t i = 0; i < moves.size(); i++) {
    int transform = TransformForPosition(input_format, history);
    // If real v6 data, can confirm that played_idx matches the inferred move.
    if (fileContents[i].visits > 0) {
      if (fileContents[i].played_idx != MoveToNNIndex(moves[i], transform)) {
        throw Exception("Move performed is not listed as played.");
      }
    }
    // Move shouldn't be marked illegal unless there is 0 visits, which should
    // only happen if invariance_info is marked with the placeholder bit.
    if (!(fileContents[i].probabilities[MoveToNNIndex(moves[i], transform)] >=
          0.0f) &&
        (fileContents[i].invariance_info & 64) == 0) {
      std::cerr << "Illegal move: " << moves[i].ToString() << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].ToString() << std::endl;
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
      int i = MoveToNNIndex(move, transform);
      int j = MoveToNNIndex(move, data->invariance_info & 7);
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

float Px0toNNUE(float q, float scaling = 416.11539129) {
  float numerator = 1 + q;
  float denominator = 1 - q;

  if (denominator == 0) {
    // Handle division by zero or return some error value
    return std::numeric_limits<float>::infinity();  // or throw an exception
  }

  return scaling * std::log(numerator / denominator);
}

struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

std::string AsNnueString(const Position& p, Move best, Move played, float q,
                         int result, const ProcessFileFlags& flags) {
  // Filter out in check and pv captures.
  static constexpr int VALUE_NONE = 32002;
  bool filtered =
      p.GetBoard().IsUnderCheck() || p.GetBoard().theirs().get(best.to());
  std::ostringstream out;
  out << "fen " << PositionToFen(p) << std::endl;
  if (p.IsBlackToMove()) best.Flip(), played.Flip();
  out << "move " << (flags.nnue_best_move ? best.ToString() : played.ToString())
      << std::endl;
  // Formula from dblue
  out << "score "
      << (filtered ? VALUE_NONE
                   : round(std::clamp(Px0toNNUE(q), -20000.0f, 20000.0f)))
      << std::endl;
  out << "ply " << p.GetGamePly() << std::endl;
  out << "result " << result << std::endl;
  out << "e" << std::endl;
  return out.str();
}

struct FileData {
  std::vector<V6TrainingData> fileContents;
  MoveList moves;
  pblczero::NetworkFormat::InputFormat input_format;
};

bool IsAllDraws(const FileData& data) {
  for (const auto& chunk : data.fileContents) {
    if (ResultForData(chunk) != 0) {
      return false;
    }
  }
  return true;
}

std::vector<V6TrainingData> ReadFile(const std::string& file) {
  std::vector<V6TrainingData> fileContents;

  TrainingDataReader reader(file);
  V6TrainingData chunk;
  while (reader.ReadChunk(&chunk)) {
    fileContents.push_back(chunk);
  }

  return fileContents;
}

FileData ProcessAndValidateFileData(std::vector<V6TrainingData> fileContents) {
  FileData data;
  data.fileContents = std::move(fileContents);

  Validate(data.fileContents);
  games += 1;
  positions += data.fileContents.size();
  // Decode moves from input data
  for (size_t i = 1; i < data.fileContents.size(); i++) {
    data.moves.push_back(
        DecodeMoveFromInput(PlanesFromTrainingData(data.fileContents[i]),
                            PlanesFromTrainingData(data.fileContents[i - 1])));
    // All moves decoded are from the point of view of the side after the
    // move so need to mirror them all to be applicable to apply to the
    // position before.
    data.moves.back().Flip();
  }
  Validate(data.fileContents, data.moves);

  data.input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
      data.fileContents[0].input_format);

  return data;
}

void ApplyPolicySubstitutions(FileData& data) {
  if (policy_subs.empty()) return;
  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;

  PopulateBoard(data.input_format, PlanesFromTrainingData(data.fileContents[0]),
                &board, &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  uint64_t rootHash = HashCat(board.Hash(), rule50ply);

  if (policy_subs.find(rootHash) != policy_subs.end()) {
    PolicySubNode* rootNode = &policy_subs[rootHash];
    for (size_t i = 0; i < data.fileContents.size(); i++) {
      if (rootNode->active) {
        for (int j = 0; j < 2062; j++) {
          data.fileContents[i].probabilities[j] = rootNode->policy[j];
        }
      }
      if (i + 1 < data.fileContents.size()) {
        int transform = TransformForPosition(data.input_format, history);
        int idx = MoveToNNIndex(data.moves[i], transform);
        if (rootNode->children[idx] == nullptr) {
          break;
        }
        rootNode = rootNode->children[idx];
        history.Append(data.moves[i]);
      }
    }
  }
}

void ApplyPolicyAdjustments(FileData& data, float distTemp, float distOffset) {
  if (distTemp == 1.0f && distOffset == 0.0f) {
    return;  // No adjustments needed
  }

  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;

  PopulateBoard(data.input_format, PlanesFromTrainingData(data.fileContents[0]),
                &board, &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  size_t move_index = 0;

  for (auto& chunk : data.fileContents) {
    const auto& board = history.Last().GetBoard();
    std::vector<bool> boost_probs(2062, false);

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
    if (move_index < data.moves.size()) {
      history.Append(data.moves[move_index]);
      move_index++;
    }
  }
}

void EstimateAndCorrectPliesLeft(FileData& data) {
  // Make move_count field plies_left for moves left head.
  int offset = 0;
  for (auto& chunk : data.fileContents) {
    // plies_left can't be 0 for real v5 data, so if it is 0 it must be a v4
    // conversion, and we should populate it ourselves with a better
    // starting estimate.
    if (chunk.plies_left == 0.0f) {
      chunk.plies_left = (int)(data.fileContents.size() - offset);
    }
    offset++;
  }
}

void ApplyDeblunder(FileData& data) {
  // Deblunder only works from v6 data onwards. We therefore check
  // the visits field which is 0 if we're dealing with upgraded data.
  if (!deblunderEnabled || data.fileContents.back().visits == 0) {
    return;
  }

  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;

  PopulateBoard(data.input_format, PlanesFromTrainingData(data.fileContents[0]),
                &board, &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);

  for (size_t i = 0; i < data.moves.size(); i++) {
    history.Append(data.moves[i]);
  }

  float activeZ[3] = {data.fileContents.back().result_q,
                      data.fileContents.back().result_d,
                      data.fileContents.back().plies_left};
  bool deblunderingStarted = false;

  while (true) {
    auto& cur = data.fileContents[history.GetLength() - 1];
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
        newZRatio = std::min(
            1.0f, (cur.best_q - cur.played_q - deblunderQBlunderThreshold) /
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
    }
    if (deblunderingStarted) {
      data.fileContents[history.GetLength() - 1].result_q = activeZ[0];
      data.fileContents[history.GetLength() - 1].result_d = activeZ[1];
      data.fileContents[history.GetLength() - 1].plies_left = activeZ[2];
    }
    if (history.GetLength() == 1) break;
    // Q values are always from the player to move.
    activeZ[0] = -activeZ[0];
    // Estimated remaining plies left has to be increased.
    activeZ[2] += 1.0f;
    history.Pop();
  }
}

void ConvertInputFormat(FileData& data, int newInputFormat) {
  if (newInputFormat == -1) return;

  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;

  PopulateBoard(data.input_format, PlanesFromTrainingData(data.fileContents[0]),
                &board, &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  ChangeInputFormat(newInputFormat, &data.fileContents[0], history);

  for (size_t i = 0; i < data.moves.size(); i++) {
    history.Append(data.moves[i]);
    ChangeInputFormat(newInputFormat, &data.fileContents[i + 1], history);
  }
}

void WriteNnueOutput(const FileData& data, const std::string& nnue_plain_file,
                     ProcessFileFlags flags) {
  // Output data in Stockfish plain format.
  if (!nnue_plain_file.empty()) {
    static Mutex mutex;
    std::ostringstream out;

    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;

    PopulateBoard(data.input_format,
                  PlanesFromTrainingData(data.fileContents[0]), &board,
                  &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);

    for (size_t i = 0; i < data.fileContents.size(); i++) {
      auto chunk = data.fileContents[i];
      Position p = history.Last();
      if (chunk.visits > 0) {
        // Format is v6 and position is evaluated.
        Move best = MoveFromNNIndex(
            chunk.best_idx, TransformForPosition(data.input_format, history));
        Move played = MoveFromNNIndex(
            chunk.played_idx, TransformForPosition(data.input_format, history));
        float q = flags.nnue_best_score ? chunk.best_q : chunk.played_q;
        out << AsNnueString(p, best, played, q, round(chunk.result_q), flags);
      } else if (i < data.moves.size()) {
        out << AsNnueString(p, data.moves[i], data.moves[i], chunk.best_q,
                            round(chunk.result_q), flags);
      }
      if (i < data.moves.size()) {
        history.Append(data.moves[i]);
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
}

void WriteOutputs(const FileData& data, const std::string& file,
                  const std::string& outputDir) {
  // Write processed training data
  if (!outputDir.empty()) {
    std::string fileName = file.substr(file.find_last_of("/\\") + 1);
    TrainingDataWriter writer(outputDir + "/" + fileName);
    for (const auto& chunk : data.fileContents) {
      // Don't save chunks that just provide move history.
      if ((chunk.invariance_info & 64) == 0) {
        writer.WriteChunk(chunk);
      }
    }
  }
}

FileData ProcessFileInternal(std::vector<V6TrainingData> fileContents,
                             float distTemp, float distOffset, int newInputFormat) {
  // Process and validate file data
  FileData data = ProcessAndValidateFileData(std::move(fileContents));

  // Apply policy substitutions if available
  ApplyPolicySubstitutions(data);

  // Apply policy adjustments (temperature, offset)
  ApplyPolicyAdjustments(data, distTemp, distOffset);

  // Estimate and correct plies left
  EstimateAndCorrectPliesLeft(data);

  // Apply deblunder processing
  ApplyDeblunder(data);

  // Convert input format if needed
  ConvertInputFormat(data, newInputFormat);

  return data;
}

void ProcessFile(const std::string& file, std::string outputDir, float distTemp,
                 float distOffset, int newInputFormat,
                 std::string nnue_plain_file, ProcessFileFlags flags) {
  try {
    // Read file data
    std::vector<V6TrainingData> fileContents = ReadFile(file);

    FileData data = ProcessFileInternal(std::move(fileContents), distTemp,
                                        distOffset, newInputFormat);

    // Write NNUE output
    WriteNnueOutput(data, nnue_plain_file, flags);

    // Write outputs
    WriteOutputs(data, file, outputDir);

  } catch (Exception& ex) {
    std::cerr << "While processing: " << file
              << " - Exception thrown: " << ex.what() << std::endl;
    if (flags.delete_files) {
      std::cerr << "It will be deleted." << std::endl;
    }
  }
  if (flags.delete_files) {
    remove(file.c_str());
  }
}

void ProcessFiles(const std::vector<std::string>& files, std::string outputDir,
                  float distTemp, float distOffset, int newInputFormat,
                  int offset, int mod, std::string nnue_plain_file,
                  ProcessFileFlags flags) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (size_t i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], outputDir, distTemp, distOffset, newInputFormat,
                nnue_plain_file, flags);
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
    for (size_t i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                              PlanesFromTrainingData(fileContents[i - 1])));
      // All moves decoded are from the point of view of the side after the
      // move so need to mirror them all to be applicable to apply to the
      // position before.
      moves.back().Flip();
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
    for (size_t i = 0; i < fileContents.size(); i++) {
      if ((fileContents[i].invariance_info & 64) == 0) {
        rootNode->active = true;
        for (int j = 0; j < 2062; j++) {
          rootNode->policy[j] = fileContents[i].probabilities[j];
        }
      }
      if (i < fileContents.size() - 1) {
        int transform = TransformForPosition(input_format, history);
        int idx = MoveToNNIndex(moves[i], transform);
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

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void RunRescorer() {
  OptionsParser options;
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  options.Add<StringOption>(kInputDirId);
  options.Add<StringOption>(kOutputDirId);
  options.Add<StringOption>(kPolicySubsDirId);
  options.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options.Add<IntOption>(kNewInputFormatId, -1, 256) = -1;
  options.Add<BoolOption>(kDeblunder) = false;
  options.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options.Add<StringOption>(kNnuePlainFileId);
  options.Add<BoolOption>(kNnueBestScoreId) = true;
  options.Add<BoolOption>(kNnueBestMoveId) = false;
  options.Add<BoolOption>(kDeleteFilesId) = true;

  if (!options.ProcessAllFlags()) return;

  if (options.GetOptionsDict().IsDefault<std::string>(kOutputDirId) &&
      options.GetOptionsDict().IsDefault<std::string>(kNnuePlainFileId)) {
    std::cerr << "Must provide an output dir or NNUE plain file." << std::endl;
    return;
  }

  if (options.GetOptionsDict().Get<bool>(kDeblunder)) {
    RescorerDeblunderSetup(
        options.GetOptionsDict().Get<float>(kDeblunderQBlunderThreshold),
        options.GetOptionsDict().Get<float>(kDeblunderQBlunderWidth));
  }

  RescorerPolicySubstitutionSetup(
      options.GetOptionsDict().Get<std::string>(kPolicySubsDirId));

  auto inputDir = options.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.empty()) {
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.empty()) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  std::transform(
      files.begin(), files.end(), files.begin(),
      [&inputDir](const std::string& file) { return inputDir + "/" + file; });
  unsigned int threads = options.GetOptionsDict().Get<int>(kThreadsId);
  ProcessFileFlags flags;
  flags.delete_files = options.GetOptionsDict().Get<bool>(kDeleteFilesId);
  flags.nnue_best_score = options.GetOptionsDict().Get<bool>(kNnueBestScoreId);
  flags.nnue_best_move = options.GetOptionsDict().Get<bool>(kNnueBestMoveId);
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([&options, offset_val, files, threads, flags]() {
        ProcessFiles(
            files, options.GetOptionsDict().Get<std::string>(kOutputDirId),
            options.GetOptionsDict().Get<float>(kTempId),
            options.GetOptionsDict().Get<float>(kDistributionOffsetId),
            options.GetOptionsDict().Get<int>(kNewInputFormatId), offset_val,
            threads,
            options.GetOptionsDict().Get<std::string>(kNnuePlainFileId), flags);
      });
    }
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(files, options.GetOptionsDict().Get<std::string>(kOutputDirId),
                 options.GetOptionsDict().Get<float>(kTempId),
                 options.GetOptionsDict().Get<float>(kDistributionOffsetId),
                 options.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
                 options.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
                 flags);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Blunders picked up by deblunder threshold: " << blunders
            << std::endl;
  std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
            << " W: " << orig_counts[2] << std::endl;
  std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
            << " W: " << fixed_counts[2] << std::endl;
}

std::vector<V6TrainingData> RescoreTrainingData(
    std::vector<V6TrainingData> fileContents, float distTemp, float distOffset,
    int newInputFormat) {
  FileData data = ProcessFileInternal(std::move(fileContents), distTemp,
                                      distOffset, newInputFormat);
  return data.fileContents;
}

bool RescorerDeblunderSetup(float threshold, float width) {
  deblunderEnabled = true;
  deblunderQBlunderThreshold = threshold;
  deblunderQBlunderWidth = width;
  return true;
}

bool RescorerPolicySubstitutionSetup(std::string policySubsDir) {
  if (!policySubsDir.empty()) {
    auto policySubFiles = GetFileList(policySubsDir);
    std::transform(policySubFiles.begin(), policySubFiles.end(),
                   policySubFiles.begin(),
                   [&policySubsDir](const std::string& file) {
                     return policySubsDir + "/" + file;
                   });
    BuildSubs(policySubFiles);
  }
  return !policy_subs.empty();
}

}  // namespace lczero