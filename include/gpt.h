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
    torch::Tensor operator()(const torch::Tensor& x);
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
    torch::Tensor operator()(const torch::Tensor& x);
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
    torch::Tensor operator()(const torch::Tensor& x);
};

// Transformer block.
class Block : public torch::nn::Module {
private:
    MultiHeadAttention self_attention;
    FeedForward feed_forward;
    torch::nn::LayerNorm layer_norm_1;
    torch::nn::LayerNorm layer_norm_2;
public:
    Block(const unsigned int& num_heads);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor operator()(const torch::Tensor& x);
};

// Decoder-only transformer model.
class GPT : public torch::nn::Module {
private:
    torch::nn::Embedding token_embedding_table;
    torch::nn::Embedding position_embedding_table;
    torch::nn::Sequential blocks;
    torch::nn::LayerNorm layer_norm;
    torch::nn::Linear lm_head;
    static void init_weights(torch::nn::Module module);
public:
    GPT(const unsigned int& vocab_size);
    torch::Tensor forward(const torch::Tensor& x);
    torch::Tensor operator()(const torch::Tensor& x);
};