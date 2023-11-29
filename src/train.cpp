#include <iostream>
// third party includes
#include "sentencepiece_trainer.h"
// local includes
#include "argparse.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " gpt_cpp --train <training data> --eval <eval data> \n[--epochs <epochs>] [--tokenizer <tokenizer model>] [--save-checkpoint <checkpoint file>] \n[--load-checkpoint <checkpoint file>] [--checkpoint-interval <interval>] \n[--eval-interval <interval>] [--eval-iters <iters>] [--learning-rate <alpha>]" << std::endl;
        return 1;
    }
    // Parse the user command line arguments to construct a TrainingConfig.
    const TrainingConfig& cfg = parse_args(argc, argv);

    // Train a tokenizer for this dataset (training and eval data) if it does not exist.
    const auto status = tokenizer::train(cfg, 2000);
    std::cout << status.ToString() << std::endl;
    if (!status.ok()) {
        std::cerr << "Error training sentencepiece tokenizer model: " << status.ToString() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Load the tokenizer model into a processor.
    sentencepiece::SentencePieceProcessor processor;
    const auto load_status = processor.Load("tok.model");
    if (!load_status.ok()) {
        std::cerr << "Error loading sentencepiece tokenizer model: " << load_status.ToString() << std::endl;
    }

    // Tokenize training and eval files.
    const auto p1_status = tokenizer::process(processor, cfg.train_file);
    if (!p1_status.ok()) {
        std::cerr << "Error processing training file: " << p1_status.ToString() << std::endl;
    }
    const auto p2_status = tokenizer::process(processor, cfg.eval_file);
    if (!p2_status.ok()) {
        std::cerr << "Error processing eval file: " << p2_status.ToString() << std::endl;
    }
    return 0;
}
