#include <iostream>
#include <utility>
#include "torch/torch.h"
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
    torch::Tensor wei = torch::matmul(q, k.transpose(1,2)) * std::pow(C, 0.5);
    torch::Tensor tril = named_buffers()["tril"];

    // (B,T,T)
    wei = wei.masked_fill(tril.slice(0, 0, T).slice(1,0,T) == 0, -INFINITY);
    wei = torch::nn::functional::softmax(wei, torch::nn::functional::SoftmaxFuncOptions(-1));
    wei = dropout(wei);

    // Perform weighted aggregation of values.
    torch::Tensor v = value(x);
    return torch::matmul(wei, v);
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
    torch::Tensor out = torch::cat(outputs, /*dim=*/2);
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
GPT::GPT(const unsigned int& vocab_size, const std::string& device) :
    device(device),
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
void GPT::init_weights(const torch::nn::Module& module) {
    if (typeid(module) == typeid(torch::nn::Linear)) {
        torch::nn::Linear linear = dynamic_cast<const torch::nn::Linear&>(module);
        torch::nn::init::normal_(linear->weight, 0.0, 0.02);
        if (linear->options.bias()) {
            torch::nn::init::zeros_(linear->bias);
        }
    } else if (typeid(module) == typeid(torch::nn::Embedding)) {
        torch::nn::Embedding embedding = dynamic_cast<const torch::nn::Embedding&>(module);
        torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
    }
}

// forward performs a forward pass through the transformer model.
// Returns the logits and the loss (if targets were specified).
// idx = 2-dimensional index into the token + position embedding tables
//       (i.e. for batch B at token T, what is the embedding vector?)
// targets = "y values" / truth, used for training but not generation.
std::pair<torch::Tensor, torch::Tensor> GPT::forward(const torch::Tensor& idx, c10::optional<torch::Tensor> labels) {
    if (idx.dim() != 2) {
        throw std::invalid_argument("input shape must be 2 dimensions");
    }
    torch::IntArrayRef sizes = idx.sizes();
    const int T = sizes[1]; // time/token dimension

    // (B,T,C)
    const torch::Tensor tok_emb = token_embedding_table(idx);

    // (T, C)
    const torch::Tensor pos_emb = position_embedding_table(torch::arange(torch::Scalar(T), torch::TensorOptions(device)));

    torch::Tensor x = tok_emb + pos_emb;     // (B,T,C)
    x = blocks->forward(x);               // (B,T,C)
    x = layer_norm(x);                 // (B,T,C)
    torch::Tensor logits = lm_head(x); // (B,T,C)

    if (!labels.has_value()) {
        // We don't calculate loss for generation (no y-values / labels).
        return {logits, torch::Tensor{nullptr}};
    }
    torch::Tensor labels_tensor = labels.value();

    if (logits.dim() != 3) {
        throw std::runtime_error("unexpected shape, logits should have 3 dimensions");
    }

    // Calculate loss.
    torch::IntArrayRef logits_sizes = logits.sizes();
    const int logits_B = logits_sizes[0]; // logits batch dimension
    const int logits_T = logits_sizes[1]; // logits token dimension
    const int logits_C = logits_sizes[2]; // logits channels
    logits = logits.view({logits_B * logits_T, logits_C}); // (B*T, C)
    const torch::Tensor y_values = labels_tensor.view({logits_B * logits_T}); // (B,T)
    const torch::Tensor loss = torch::nn::functional::cross_entropy(logits, y_values.to(torch::kLong));
    return {logits, loss};
}

std::pair<torch::Tensor, torch::Tensor> GPT::operator()(const torch::Tensor& idx, c10::optional<torch::Tensor> labels) {
    return forward(idx, labels);
}

// generate predicts the next `max_new_tokens` given the input idx, which an index
// of shape (B,T) representing the current context.
torch::Tensor GPT::generate(torch::Tensor& idx, const unsigned int& max_new_tokens) {
    for (int i = 0; i < max_new_tokens; ++i) {
        // crop context to last "max sequence length" tokens
        torch::IntArrayRef idx_sizes = idx.sizes();
        unsigned int ctx_len = idx_sizes[1];
        torch::Tensor idx_cond = idx.slice(/*dim=*/0).slice(1, /*start=*/ctx_len-SEQ_LEN);

        // predict next token
        auto[logits, loss] = forward(idx);

        // focus only on the last step in time (final token)
        torch::IntArrayRef logits_sizes = logits.sizes();

        // logits = logits[:, -1, :]
        logits = logits.slice(1, logits.size(1) - 1, logits.size(1)).squeeze(1);

        // apply softmax to get probabilities (along token dimension)
        torch::Tensor probs = torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(1));

        // sample from that probability distribution
        torch::Tensor next_idx = torch::multinomial(probs, /*num_samples*/1);

        // add the new index to the context for the next iteration
        // by concatenating the predicted tokens for each batch along the token dimension.
        idx = torch::cat({idx, next_idx}, /*dim=*/1);
    }
    return idx;
}