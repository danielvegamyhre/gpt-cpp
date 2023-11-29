#include <fstream>
// third party includes
#include "torch/torch.h"
#include "sentencepiece_trainer.h"
// local includes
#include "train.h"
#include "tokenizer.h"

// The prefix of the tokenizer model file. SentencePiece will automatically
// save the tokenizer model to a file with the name format "<prefix>.model"
// (e.g. "tok.model").
#define TOKENIZER_MODEL_PREFIX "tok"
#define TOKENIZED_FILE_SUFFIX "_tokenized"

namespace tokenizer {
    Status train(const TrainingConfig& cfg, const unsigned int &vocab_size) {
        char args[500];
        char *format = "--input=%s,%s --model_prefix=%s --vocab_size=%d --input_sentence_size=200 --shuffle_input_sentence=true";
        std::printf(format, cfg.train_file.data(), cfg.eval_file.data(), TOKENIZER_MODEL_PREFIX, vocab_size);
        std::snprintf(args, 500, format, cfg.train_file.data(), cfg.eval_file.data(), TOKENIZER_MODEL_PREFIX, vocab_size);
        return sentencepiece::SentencePieceTrainer::Train(args);
    }

    Status process(const sentencepiece::SentencePieceProcessor& processor, const std::string& input_file) {
        std::ifstream input(input_file);
        if (!input.is_open()) {
            std::cerr << "Unable to input file: " << input_file << std::endl;
            return {sentencepiece::util::StatusCode::kPermissionDenied, "Unable to open input file."};
        }

        // Read input file into token
        std::string line, data;
        while (std::getline(input, line)) {
            data += line + "\n";
        }
        input.close();

        std::cout << data << std::endl;

        // Tokenize the data and store it in numeric/tensor format in the output file.
        std::cout << "TOKENIZING INPUT FILE: " << input_file << std::endl;
        std::vector<int> ids = processor.EncodeAsIds(data);
        torch::Tensor tensor = torch::tensor(ids);

        std::cout << "TENSOR NUM ELEMS: " << tensor.numel() << std::endl;
        torch::save(tensor, input_file + TOKENIZED_FILE_SUFFIX);
        return {sentencepiece::util::StatusCode::kOk, "Finished tokenizing input file and saved tensor output file."};
    }
} // tokenizer namespace