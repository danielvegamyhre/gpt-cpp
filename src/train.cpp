#include <iostream>
// third party includes
#include "sentencepiece_trainer.h"
// local includes
#include "argparse.h"
#include "gpt.h"

// DATA maps a split (train, eval) to the tensor representation of the data.
static std::unordered_map<std::string, torch::Tensor> DATA;

// get_batch returns a pair of tensors (X,Y) where X is data from the given split (train or eval),
// and Y is the corresponding labels.
std::pair<torch::Tensor, torch::Tensor> get_batch(const std::string& split, const std::string& device) {
    const torch::Tensor data = DATA[split];
    // Get BATCH_SIZE random indexes between 0 and len(data)-SEQ_LEN (since we will be using the following
    // SEQ_LEN elements after the index).
    torch::Tensor ix = torch::randint(data.size(0) - SEQ_LEN, {BATCH_SIZE});
    std::vector<torch::Tensor> x_vec, y_vec;
    for (int i = 0; i < ix.size(0); i++) {
        // input sequence.
        x_vec.push_back(data.slice(0, i, i+SEQ_LEN));

        // labels are the next index for each index in the input sequence,
        // so our model can predict the next token for each index in x, and
        // compare to the ground truth in y.
        y_vec.push_back(data.slice(0, i+1, i+1+SEQ_LEN));
    }
    torch::Tensor x = torch::stack(x_vec);
    torch::Tensor y = torch::stack(y_vec);
    x.to(device);
    y.to(device);
    return {x, y};
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " gpt_cpp --train <training data> --eval <eval data> \n[--epochs <epochs>] [--tokenizer <tokenizer model>] [--save-checkpoint <checkpoint file>] \n[--load-checkpoint <checkpoint file>] [--checkpoint-interval <interval>] \n[--eval-interval <interval>] [--eval-iters <iters>] [--learning-rate <alpha>]" << std::endl;
        return 1;
    }
    // Parse the user command line arguments to construct a TrainingConfig.
    const TrainingConfig& cfg = parse_args(argc, argv);

    // Train a tokenizer for this dataset (training and eval data) if it does not exist.
    const auto status = tokenizer::train(cfg, 646);
    std::cout << status.ToString() << std::endl;
    if (!status.ok()) {
        std::cerr << "Error training sentencepiece tokenizer model: " << status.ToString() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Load the tokenizer model into a processor.
    sentencepiece::SentencePieceProcessor processor;
    const auto load_status = processor.Load("tok.model");
    if (!load_status.ok()) {
        std::cerr << "Error loading sentencepiece tokenizer model: " << load_status.ToString() << std::endl;
    }

    // Tokenize training and eval files.
    const auto [train_tensor, p1_status] = tokenizer::process(processor, cfg.train_file);
    if (p1_status.code() != sentencepiece::util::StatusCode::kOk) {
        std::cerr << "Error processing training file: " << p1_status.ToString() << std::endl;
    }
    std::cout << "Finished tokenizing training file: " << cfg.train_file << std::endl;

    const auto [eval_tensor, p2_status] = tokenizer::process(processor, cfg.eval_file);
    if (p2_status.code() != sentencepiece::util::StatusCode::kOk) {
        std::cerr << "Error processing eval file: " << p2_status.ToString() << std::endl;
    }
    std::cout << "Finished tokenizing eval file: " << cfg.eval_file << std::endl;

    // Store our train/eval data in memory.
    DATA[TRAIN] = train_tensor;
    DATA[EVAL] = eval_tensor;

    // Initialize our model.
    GPT model = GPT(cfg.vocab_size, cfg.device);
    model.to(cfg.device);

    // Create optimizer.
    torch::optim::AdamWOptions opts = {/*lr=*/cfg.learning_rate};
    torch::optim::AdamW optim = torch::optim::AdamW(model.parameters(), opts);

    // Load model checkpoint if specified.
    if (cfg.load_checkpoint.has_value()) {
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(cfg.load_checkpoint.value());
        model.load(input_archive);
    }
    uint32_t start_epoch = 0;

    // Generate output if this is not a training run.
    if (cfg.generate > 0) {
        std::cout << "Generating output of length: " << cfg.generate << std::endl;
        auto ctx = torch::zeros({1,1}, torch::kLong).to({cfg.device});
        auto generate = [&](const uint32_t max_new_tokens){
            const torch::Tensor output_token_ids = model.generate(ctx, max_new_tokens);
            return tokenizer::decode(processor, output_token_ids);
        };
        std::cout << generate(cfg.generate) << std::endl;
        return EXIT_SUCCESS;
    }

    // Training loop
    std::cout << "Starting training" << std::endl;
    for (int epoch = start_epoch; epoch < cfg.epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        // Don't estimate loss on first epoch.
        if (epoch != start_epoch && epoch % cfg.eval_interval == 0) {
            std::cout << "Estimating loss on eval data" << std::endl;
            // TODO: estimate loss
        }

        // Get batch of training data + labels.
        auto [xb, yb] = get_batch(TRAIN, cfg.device);

        // Perform forward pass.
        auto [_, loss] = model.forward(xb, yb);

        // Zero out gradients.
        optim.zero_grad();

        // Backward pass.
        loss.backward();

        // Update parameters.
        optim.step();

        // Checkpoint model.
        if (epoch != start_epoch) {
            // If this epoch is a checkpoint interval, or we are on the last epoch, save the model.
            if (cfg.checkpoint_interval.has_value() && epoch % cfg.checkpoint_interval.value() == 0 || epoch == cfg.epochs - 1) {
                std::cout << "Checkpointing model" << std::endl;

                // Serialize model to output archive.
                torch::serialize::OutputArchive output_archive;
                model.save(output_archive);

                // Write serialized model to disk.
                std::string model_path = "model.pt";
                if (cfg.save_checkpoint.has_value()) {
                    model_path = cfg.save_checkpoint.value();
                }
                output_archive.save_to(model_path);
                std::cout << "Checkpointed model to: " << model_path << std::endl;
            }
        }
    }

    std::cout << "Training complete." << std::endl;
    return EXIT_SUCCESS;
}
