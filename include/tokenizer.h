#pragma once
// third party includes
#include "sentencepiece_trainer.h"
// local includes
#include "train.h"

using Status = sentencepiece::util::Status;

namespace tokenizer {
    // train trains a sentencepiece tokenizer on the training and eval datasets defined in
    // the given TrainingConfig, and returns a Status indicating the result.
    Status train(const TrainingConfig& cfg, const unsigned int &vocab_size);

    // process tokenizes an input file into a numeric (tensor) format and stores the output in an
    // output file, which has the same file path as the input but with a "_tokenized" suffix appended.
    // Returns a Status object indicating the result.
    Status process(const sentencepiece::SentencePieceProcessor& processor, const std::string& input_file);
}