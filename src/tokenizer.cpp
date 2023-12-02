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
        std::snprintf(args, 500, format, cfg.train_file.data(), cfg.eval_file.data(), TOKENIZER_MODEL_PREFIX, vocab_size);
        std::cout << "training sentencepiece tokenizer model with args: " << args << std::endl;
        return sentencepiece::SentencePieceTrainer::Train(args);
    }

    std::pair<torch::Tensor*, Status> process(const sentencepiece::SentencePieceProcessor& processor, const std::string& input_file) {
        std::ifstream input(input_file);
        if (!input.is_open()) {
            std::cerr << "Unable to input file: " << input_file << std::endl;
            return {nullptr, {sentencepiece::util::StatusCode::kPermissionDenied, "Unable to open input file."}};
        }

        // Read input file into token
        std::string line, data;
        while (std::getline(input, line)) {
            data += line + "\n";
        }
        input.close();

        // Tokenize the data and store it in numeric/tensor format in the output file.
        std::vector<int> ids = processor.EncodeAsIds(data);
        torch::Tensor tensor = torch::tensor(ids, torch::kInt);

        torch::save(tensor, input_file + TOKENIZED_FILE_SUFFIX);
        return {&tensor, {sentencepiece::util::StatusCode::kOk, "Finished tokenizing input file and saved tensor output file."}};
    }

    std::string decode(const sentencepiece::SentencePieceProcessor& processor, const torch::Tensor& ids) {
        std::string text;
        torch::Tensor int_ids = ids.to(torch::kInt);
        std::vector<int> ids_vec = std::vector<int>(int_ids.data_ptr<int>(), int_ids.data_ptr<int>() + int_ids.numel());
        processor.Decode(ids_vec, &text);
        return text;
    }
} // tokenizer namespace