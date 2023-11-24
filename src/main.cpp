#include <iostream>
#include <torch/torch.h>
#include "gpt.h"


int main() {
    torch::Tensor x = torch::rand({2, 384});
    std::cout << x << std::endl;

    FeedForward ff = FeedForward(EMBED_SIZE);
    torch::Tensor out = ff.forward(x);

    std::cout << out << std::endl;
}