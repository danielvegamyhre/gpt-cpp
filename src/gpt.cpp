#include <iostream>
#include <torch/torch.h>
#include "gpt.h"

// FeedForward implementation.
FeedForward::FeedForward(const unsigned int& num_embed_dims) :
    seq(torch::nn::Linear(num_embed_dims, 4*num_embed_dims),
            torch::nn::ReLU(),
            torch::nn::Linear(4*num_embed_dims, num_embed_dims),
            torch::nn::Dropout(DROPOUT)) {
    register_module("seq", seq);
};

torch::Tensor FeedForward::forward(const torch::Tensor& x) {
    return seq->forward(x);
};

torch::Tensor FeedForward::operator()(const torch::Tensor& x) {
    return forward(x);
}

// Head implementation.
Head::Head(const unsigned int& head_size) :
    head_size(head_size),
    key(torch::nn::Linear(torch::nn::LinearOptions(EMBED_SIZE, head_size).bias(false))),
    query(torch::nn::Linear(torch::nn::LinearOptions(EMBED_SIZE, head_size).bias(false))),
    value(torch::nn::Linear(torch::nn::LinearOptions(EMBED_SIZE, head_size).bias(false))),
    dropout(torch::nn::Dropout(DROPOUT)) {

    register_module("key", key);
    register_module("query", query);
    register_module("value", value);
    register_module("dropout", dropout);

    // Create a lower triangular matrix and register it as a buffer to use in self-attention.
    torch::Tensor tril = torch::tril(torch::ones({SEQ_LEN, SEQ_LEN}));
    register_buffer("tril", tril);
}

// forward accepts a tensor of dimensions (B,T,C) where:
// B = batch size
// T = time dimension
// C = channels
torch::Tensor Head::forward(const torch::Tensor& x) {
    if (x.dim() != 3) {
        throw std::invalid_argument("input tensor must be 3 dimensional");
    }
    torch::IntArrayRef sizes = x.sizes();
    const unsigned int B = sizes[0];
    const unsigned int T = sizes[1];
    const unsigned int C = sizes[2];

    torch::Tensor k = key(x);   // (B, T, head_size)
    torch::Tensor q = query(x); // (B, T, head_size)

    // (B,T,head_size) * (B,head_size,T) = (B,T,T)
    torch::Tensor wei = torch::mm(q, k.transpose(-2, -1)) * std::pow(C, 0.5);
    torch::Tensor tril = named_buffers()["tril"];

    // (B,T,T)
    wei = wei.masked_fill(tril.slice(0, 0, T).slice(1,0,T) == 0, -INFINITY);
    wei = torch::nn::functional::softmax(wei, torch::nn::functional::SoftmaxFuncOptions(-1));
    wei = dropout(wei);

    // Perform weighted aggregation of values.
    torch::Tensor v = value(x);
    return torch::mm(wei, v);
}

torch::Tensor Head::operator()(const torch::Tensor& x) {
    return forward(x);
}

// MultiHeadAttention implementation.
MultiHeadAttention::MultiHeadAttention(const unsigned int &num_heads, const unsigned int &head_size) :
    heads(torch::nn::ModuleList()),
    projection(torch::nn::Linear(num_heads * head_size, EMBED_SIZE)),
    dropout(torch::nn::Dropout(DROPOUT)) {

    for (int i=0; i < num_heads; ++i) {
        heads->push_back(Head(head_size));
    }
}

torch::Tensor MultiHeadAttention::forward(const torch::Tensor& x) {
    std::vector<torch::Tensor> outputs;
    for (const auto&  module : *heads) {
        outputs.push_back(module->as<Head>()->forward(x));
    }
    torch::Tensor out = torch::cat(outputs);
    out = projection(out);
    return dropout(out);
}

torch::Tensor MultiHeadAttention::operator()(const torch::Tensor& x) {
    return forward(x);
}

// Transformer block implementation.
Block::Block(const unsigned int& num_heads) :
        self_attention(MultiHeadAttention(num_heads, EMBED_SIZE/num_heads)),
        feed_forward(EMBED_SIZE),
        layer_norm_1(torch::nn::LayerNorm(torch::nn::LayerNormOptions({EMBED_SIZE}))),
        layer_norm_2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({EMBED_SIZE}))) {
}

torch::Tensor Block::forward(torch::Tensor x) {
    x = x + self_attention(layer_norm_1(x));
    x = x + feed_forward(layer_norm_2(x));
    return x;
}

torch::Tensor Block::operator()(const torch::Tensor& x) {
    return forward(x);
}


// Decoder-only transformer model implementation.
GPT::GPT(const unsigned int& vocab_size) :
    token_embedding_table(torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, EMBED_SIZE))),
    position_embedding_table(torch::nn::Embedding(torch::nn::EmbeddingOptions(SEQ_LEN, EMBED_SIZE))),
    blocks(torch::nn::Sequential(
            Block(4),
            Block(4),
            Block(4))),
    layer_norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({EMBED_SIZE}))),
    lm_head(torch::nn::Linear(EMBED_SIZE, vocab_size)) {

    apply(init_weights);
}

// Weight initialization
void GPT::init_weights(torch::nn::Module module) {

}