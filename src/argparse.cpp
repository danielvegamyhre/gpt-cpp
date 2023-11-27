#include <iostream>
#include <argparse.h>

TrainingConfig& parse_args(int argc, char* argv[]) {
    static TrainingConfig cfg;
    // argv[0] is the program name, so we start at index 1.
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];

        // Check if this is a no-arg flag.
        if (auto flag = NoArgs.find(opt); flag != NoArgs.end()) {
            flag->second(cfg);
        }

        // Check if this is a one-arg flag.
        else if (auto flag = OneArgs.find(opt); flag != OneArgs.end()) {
            // Check if the arg is actually specified.
            if (++i < argc) {
                flag->second(cfg, argv[i]);
            } else {
                throw std::runtime_error("missing param after: " + opt);
            }
        }

        // Throw runtime errors for unrecognized arguments.
        else {
            throw std::runtime_error("unrecognized flag: " + opt);
        }
    }
    return cfg;
}