#include <iostream>
// third party includes
#include <sentencepiece_trainer.h>
// local includes
#include <train.h>

namespace tokenizer {
    int train(const TrainingConfig& cfg, const unsigned int &vocab_size) {
        char args[500];
        char *format = "--input=%s,%s --model_prefix=m --vocab_size=%d";

        std::snprintf(args, 500, format, cfg.train_file.data(), cfg.eval_file.data(), vocab_size);

        auto status = sentencepiece::SentencePieceTrainer::Train(args);
        std::cout << status.ToString() << std::endl;
        return 0;
    }
} // tokenizer namespace