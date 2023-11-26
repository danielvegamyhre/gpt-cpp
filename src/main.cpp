#include <iostream>
#include <torch/torch.h>
#include "gpt.h"

// learning rate (alpha)
static const float LEARNING_RATE = 1e-3;
static const std::string DEVICE = "mps";
static const int EVAL_INTERVAL = 1000;
static const int EVAL_ITERS = 100;
static const std::string TRAIN = "train";
static const std::string EVAL = "eval";

int main() {
    torch::Tensor x = torch::rand({2, 384});
    std::cout << x << std::endl;

    FeedForward ff = FeedForward(EMBED_SIZE);
    torch::Tensor out = ff.forward(x);

    std::cout << out << std::endl;
}