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

#include "rescorer/rescoreloop.h"

#include <optional>
#include <regex>
#include <sstream>

#include "neural/decoder.h"
#include "trainingdata/reader.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"

#ifdef _WIN64
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

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
const OptionId kNnueEvaluatorId{
    "nnue-evaluator", "", "Use NNUE evaluator to rescore the training data."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};

class NNUEEvaluator {
 public:
  NNUEEvaluator(const std::string& evaluator) {
#ifdef _WIN64
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&childStdInRead, &childStdInWrite, &saAttr, 0)) {
      throw Exception("StdIn pipe creation failed");
    }
    if (!SetHandleInformation(childStdInWrite, HANDLE_FLAG_INHERIT, 0)) {
      throw Exception("StdIn pipe set handle information failed");
    }

    if (!CreatePipe(&childStdOutRead, &childStdOutWrite, &saAttr, 0)) {
      throw Exception("StdOut pipe creation failed");
    }
    if (!SetHandleInformation(childStdOutRead, HANDLE_FLAG_INHERIT, 0)) {
      throw Exception("StdOut pipe set handle information failed");
    }

    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));
    ZeroMemory(&si, sizeof(STARTUPINFO));
    si.cb = sizeof(STARTUPINFO);
    si.hStdError = childStdOutWrite;
    si.hStdOutput = childStdOutWrite;
    si.hStdInput = childStdInRead;
    si.dwFlags |= STARTF_USESTDHANDLES;

    if (!CreateProcess(NULL, const_cast<LPSTR>(evaluator.c_str()), NULL, NULL, TRUE, 0,
                       NULL, NULL, &si, &pi)) {
      throw Exception("Create process failed");
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(childStdOutWrite);
    CloseHandle(childStdInRead);
#else
    if (pipe(pipe_in) == -1 || pipe(pipe_out) == -1) {
      throw Exception("Failed to create pipes.");
    }

    pid = fork();
    if (pid == -1) {
      throw Exception("Failed to fork.");
    }

    if (pid == 0) {
      dup2(pipe_in[0], STDIN_FILENO);
      dup2(pipe_out[1], STDOUT_FILENO);

      close(pipe_in[0]);
      close(pipe_in[1]);
      close(pipe_out[0]);
      close(pipe_out[1]);

      char* argv[] = {nullptr};
      execve(evaluator.c_str(), argv, environ);
      std::cerr << "Create process failed." << std::endl;
      _exit(1);
    } else {
      close(pipe_in[0]);
      close(pipe_out[1]);
    }
#endif
  }

  ~NNUEEvaluator() {
    std::string command = "quit\n";
#ifdef _WIN64
    DWORD written;
    WriteFile(childStdInWrite, command.c_str(), command.length(), &written, NULL);
    CloseHandle(childStdInWrite);
    CloseHandle(childStdOutRead);
#else
    [[maybe_unused]] int written = write(pipe_in[1], command.c_str(), command.length());
    if (pid != 0) {
      close(pipe_in[1]);
      close(pipe_out[0]);
      waitpid(pid, nullptr, 0);
    }
#endif
  }

  std::pair<float, float> EvaluatePosition(const std::string& fen) {
    std::string command = "fen " + fen + "\neval\n";
#ifdef _WIN64
    DWORD written;
    if (!WriteFile(childStdInWrite, command.c_str(), command.length(), &written,
                   NULL)) {
#else
    if (write(pipe_in[1], command.c_str(), command.length()) < 0) {
#endif
      throw Exception("Failed to write to pipe");
    }

    char buffer[256];
    std::string output;
#ifdef _WIN64
    DWORD bytes_read;
    while (ReadFile(childStdOutRead, buffer, sizeof(buffer) - 1, &bytes_read,
                    NULL) &&
           bytes_read > 0) {
#else
    ssize_t bytes_read;
    while ((bytes_read = read(pipe_out[0], buffer, sizeof(buffer) - 1)) > 0) {
#endif
      buffer[bytes_read] = '\0';
      output = buffer;
      if (output.find("wdl") != std::string::npos) {
        break;
      }
    }
    std::regex re(R"(wdl\s(\d+)\s(\d+)\s(\d+))");
    std::smatch match;
    if (std::regex_search(output, match, re)) {
      float w = std::stoi(match[1]) / 1000.0;
      float d = std::stoi(match[2]) / 1000.0;
      float l = std::stoi(match[3]) / 1000.0;
      float q = w - l;
      return {q, d};
    } else {
      throw Exception("Failed to extract WDL from output.");
    }
  }

 private:
#ifdef _WIN64
  HANDLE childStdInRead = NULL;
  HANDLE childStdInWrite = NULL;
  HANDLE childStdOutRead = NULL;
  HANDLE childStdOutWrite = NULL;
#else
  int pipe_in[2];
  int pipe_out[2];
  pid_t pid;
#endif
};

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

void Validate(const std::vector<V6TrainingData>& fileContents) {
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
  for (size_t i = 0; i < moves.size(); i++) {
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

float Px0toNNUE(float q, float scaling = 416.11539129) {
  float numerator =  1 + q;
  float denominator = 1 - q;

  if (denominator == 0) {
    // Handle division by zero or return some error value
    return std::numeric_limits<float>::infinity();  // or throw an exception
  }

  return scaling * std::log(numerator / denominator);
}

std::string AsNnueString(const Position& p, Move best, Move played, float q,
                         int result, const ProcessFileFlags& flags) {
  // Filter out in check and pv captures.
  static constexpr int VALUE_NONE = 32002;
  bool filtered = p.GetBoard().IsUnderCheck() ||
                  p.GetBoard().theirs().get(best.to());
  std::ostringstream out;
  out << "fen " << GetFen(p) << std::endl;
  if (p.IsBlackToMove()) best.Mirror(), played.Mirror();
  out << "move "
      << (flags.nnue_best_move ? best.as_string() : played.as_string())
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

void ProcessFile(const std::string& file, std::string outputDir, float distTemp,
                 float distOffset, int newInputFormat,
                 std::string nnue_plain_file, ProcessFileFlags flags,
                 std::unique_ptr<NNUEEvaluator>& evaluator) {
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
      for (size_t i = 1; i < fileContents.size(); i++) {
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
        for (size_t i = 0; i < fileContents.size(); i++) {
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
          if (i + 1 < fileContents.size()) {
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
      orig_counts[ResultForData(fileContents[0]) + 1]++;
      fixed_counts[ResultForData(fileContents[0]) + 1]++;
      for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        history.Append(moves[i]);
      }

      if (distTemp != 1.0f || distOffset != 0.0f) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        int move_index = 0;
        for (auto& chunk : fileContents) {
          std::vector<bool> boost_probs(2062, false);

          float sum = 0.0;
          int prob_index = 0;
          for (auto& prob : chunk.probabilities) {
            float offset = distOffset;
            prob_index++;
            if (prob < 0 || std::isnan(prob)) continue;
            prob = std::max(0.0f, prob + offset);
            prob = std::pow(prob, 1.0f / distTemp);
            sum += prob;
          }
          prob_index = 0;
          for (auto& prob : chunk.probabilities) {
            prob_index++;
            if (prob < 0 || std::isnan(prob)) continue;
            prob /= sum;
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
        for (size_t i = 0; i < moves.size(); i++) {
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
        for (size_t i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          ChangeInputFormat(newInputFormat, &fileContents[i + 1], history);
        }
      }

      // If an NNUE evaluator is provided, use it to rescore the training data
      // and update the best_q and best_d field.
      if (evaluator) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t i = 0; i < fileContents.size(); i++) {
          auto& chunk = fileContents[i];
          if (chunk.visits > 0) {
            auto [q, d] = evaluator->EvaluatePosition(GetFen(history.Last()));
            chunk.best_q = q;
            chunk.best_d = d;
          }
          if (i < moves.size()) {
            history.Append(moves[i]);
          }
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
        for (size_t i = 0; i < fileContents.size(); i++) {
          auto chunk = fileContents[i];
          Position p = history.Last();
          if (chunk.visits > 0) {
            // Format is v6 and position is evaluated.
            Move best = MoveFromNNIndex(chunk.best_idx,
                                        TransformForPosition(format, history));
            Move played = MoveFromNNIndex(
                chunk.played_idx, TransformForPosition(format, history));
            float q = flags.nnue_best_score ? chunk.best_q : chunk.played_q;
            out << AsNnueString(p, best, played, q, round(chunk.result_q),
                                flags);
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
                  float distTemp, float distOffset, int newInputFormat,
                  int offset, int mod, std::string nnue_plain_file,
                  ProcessFileFlags flags, std::string nnue_evaluator) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  std::unique_ptr<NNUEEvaluator> evaluator;
  if (!nnue_evaluator.empty()) {
    evaluator = std::make_unique<NNUEEvaluator>(nnue_evaluator);
  }
  for (size_t i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], outputDir, distTemp, distOffset, newInputFormat,
                nnue_plain_file, flags, evaluator);
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
    for (size_t i = 0; i < fileContents.size(); i++) {
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

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void RescoreLoop::RunLoop() {
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
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
  options_.Add<StringOption>(kNnueEvaluatorId) = "";
  options_.Add<BoolOption>(kDeleteFilesId) = true;

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
    for (size_t i = 0; i < policySubFiles.size(); i++) {
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
  for (size_t i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  unsigned int threads = options_.GetOptionsDict().Get<int>(kThreadsId);
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
            files, options_.GetOptionsDict().Get<std::string>(kOutputDirId),
            options_.GetOptionsDict().Get<float>(kTempId),
            options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
            options_.GetOptionsDict().Get<int>(kNewInputFormatId), offset_val,
            threads,
            options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
            flags,
            options_.GetOptionsDict().Get<std::string>(kNnueEvaluatorId));
      });
    }
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(
        files, options_.GetOptionsDict().Get<std::string>(kOutputDirId),
        options_.GetOptionsDict().Get<float>(kTempId),
        options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
        options_.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
        options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId), flags,
        options_.GetOptionsDict().Get<std::string>(kNnueEvaluatorId));
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

}  // namespace lczero
