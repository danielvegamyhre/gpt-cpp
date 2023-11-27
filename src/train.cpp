#include <iostream>
#include <argparse.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " gpt_cpp --train <training data> --eval <eval data> \n[--epochs <epochs>] [--tokenizer <tokenizer model>] [--save-checkpoint <checkpoint file>] \n[--load-checkpoint <checkpoint file>] [--checkpoint-interval <interval>] \n[--eval-interval <interval>] [--eval-iters <iters>] [--learning-rate <alpha>]" << std::endl;
        return 1;
    }
    TrainingConfig& cfg = parse_args(argc, argv);
    tokenizer::train(cfg, 9000);
    return 0;
}
