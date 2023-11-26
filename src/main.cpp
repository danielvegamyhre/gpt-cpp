#include <iostream>
#include <tokenizer.h>

// learning rate (alpha)
static const float LEARNING_RATE = 1e-3;
static const std::string DEVICE = "mps";
static const int EVAL_INTERVAL = 1000;
static const int EVAL_ITERS = 100;
static const std::string TRAIN = "train";
static const std::string EVAL = "eval";

int main() {
    std::cout << "training tokenizer\n";
    tokenizer::train("data/shakespeare/train.txt", 10000);
    return 0;
}
