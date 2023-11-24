#pragma once
#include <iostream>
#include <torch/torch.h>

// Number of independent examples to process at once.
static const unsigned int BATCH_SIZE = 64;

// Maximum context length for one input.
static const unsigned int SEQ_LEN = 256;

// Embedding dimension size.
static const unsigned int EMBED_SIZE = 384;

// Dropout ratio.
static const float DROPOUT = 0.2f;

// Simple feed-forward network.
class FeedForward : public torch::nn::Module {
private:
    torch::nn::Sequential seq;
public:
    FeedForward(const unsigned int& num_embed_dims);
    torch::Tensor forward(const torch::Tensor& x);
};

// Single head of self-attention.
class Head : public torch::nn::Module {
private:
    const unsigned int head_size;
    torch::nn::Linear key;
    torch::nn::Linear query;
    torch::nn::Linear value;
    torch::nn::Dropout dropout;
public:
    Head(const unsigned int& head_size);
    torch::Tensor forward(const torch::Tensor& x);
};

// Multi-head attention layer of a transformer block.
class MultiHeadAttention : public torch::nn::Module {
private:
    torch::nn::ModuleList heads;
    torch::nn::Linear projection;
    torch::nn::Dropout dropout;
public:
    MultiHeadAttention(const unsigned int& num_heads, const unsigned int& head_size);
    torch::Tensor forward(const torch::Tensor& x);
};