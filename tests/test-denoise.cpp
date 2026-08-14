#include <cstdlib>
#include <iostream>
#include <vector>

#include "common/common.h"
#include "common/log.h"

namespace {

void print_usage(const char* program, const std::vector<ArgOptions>& options) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << program << " [sd-cli image-generation options]\n\n";
    std::cout << "Runs latent preparation, prompt conditioning, and all sampling steps. "
                 "It does not decode or save an image.\n";
    std::cout << "Set SD_HTP_DENOISE_STEP_TRACE=1 to print each diffusion-model forward time.\n\n";
    std::cout << "Context Options:\n";
    options[0].print();
    std::cout << "\nGeneration Options:\n";
    options[1].print();
}

void log_callback(enum sd_log_level_t level, const char* log, void*) {
    log_print(level, log, log_verbose, log_color);
}

}  // namespace

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDContextParams ctx_params;
    SDGenerationParams gen_params;
    std::vector<ArgOptions> options = {ctx_params.get_options(), gen_params.get_options()};

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0], options);
        return EXIT_SUCCESS;
    }

    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0], options);
        return EXIT_FAILURE;
    }

    if (!ctx_params.resolve_and_validate(IMG_GEN) ||
        !gen_params.resolve_and_validate(IMG_GEN,
                                         ctx_params.lora_model_dir,
                                         ctx_params.hires_upscalers_dir,
                                         true)) {
        print_usage(argv[0], options);
        return EXIT_FAILURE;
    }

    sd_set_log_callback(log_callback, nullptr);
    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(true, true, false, true);
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
    if (!sd_ctx) {
        LOG_ERROR("new_sd_ctx failed");
        return EXIT_FAILURE;
    }

    if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
    }
    if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
        gen_params.sample_params.scheduler =
            sd_get_default_scheduler(sd_ctx.get(), gen_params.sample_params.sample_method);
    }

    sd_img_gen_params_t image_params = gen_params.to_sd_img_gen_params_t();
    return denoise_image(sd_ctx.get(), &image_params) ? EXIT_SUCCESS : EXIT_FAILURE;
}
