#include <iostream>
#include <sentencepiece_trainer.h>

namespace tokenizer {
    int train(const char *input_file, const unsigned int &vocab_size) {
        char args[100];
        char *format = "--input=%s --model_prefix=m --vocab_size=%d";
        std::snprintf(args, 100, format, input_file, vocab_size);

        std::cout << "training sentencepiece model with args: " << args << "\n";

        sentencepiece::SentencePieceTrainer::Train(args);
        return 0;
    }
} // tokenizer namespace