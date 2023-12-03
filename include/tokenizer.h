#pragma once
// third party includes
#include "sentencepiece_trainer.h"
#include "torch/torch.h"
// local includes
#include "train.h"

using Status = sentencepiece::util::Status;

namespace tokenizer {
    // train trains a sentencepiece tokenizer on the training and eval datasets defined in
    // the given TrainingConfig, and returns a Status indicating the result.
    Status train(const TrainingConfig& cfg, const unsigned int &vocab_size);

    // process tokenizes an input file into a numeric (tensor) format and stores the output in an
    // output file, which has the same file path as the input but with a "_tokenized" suffix appended.
    // Returns a pair containing the tensor representation of the tokenized data, and a Status object
    // indicating the result (error or ok).
    std::pair<torch::Tensor, Status> process(const sentencepiece::SentencePieceProcessor& processor, const std::string& input_file);

    // decode accepts a tensor of token IDs and decodes them into the corresponding output text.
    std::string decode(const sentencepiece::SentencePieceProcessor& processor, const torch::Tensor& ids);
}