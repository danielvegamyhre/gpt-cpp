#include <iostream>
#include <sentencepiece_trainer.h>


int train(const std::string& input_file, const unsigned int& vocab_size) {
    std::string args = "test\n";
//    std::string args = std::format("input_file: {}, vocab_size: {}\n", input_file, vocab_size);
    std::cout << args << std::endl;
    sentencepiece::SentencePieceTrainer::Train("--input=test/botchan.txt --model_prefix=m --vocab_size=1000");
    return 0;
}