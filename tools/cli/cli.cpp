#include "arg.h"
#include "common.h"
#include "log.h"
#include "gguf.h"

#include "cli-context.h"
#include "cli-diffusion.h"

#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (cli_context::interrupted().load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    cli_context::interrupted().store(true);
}
#endif

static bool is_diffusion_model(const std::string & path) {
    if (path.empty()) {
        return false;
    }

    gguf_init_params params = { true, nullptr };
    gguf_context * metadata = gguf_init_from_file(path.c_str(), params);
    if (!metadata) {
        return false;
    }

    bool result = false;
    const int64_t key = gguf_find_key(metadata, "general.architecture");
    if (key >= 0 && gguf_get_kv_type(metadata, key) == GGUF_TYPE_STRING) {
        const std::string arch = gguf_get_val_str(metadata, key);
        result = arch == "dream" || arch == "llada" || arch == "llada-moe" ||
                 arch == "rnd1" || arch == "diffusion-gemma";
    }

    gguf_free(metadata);
    return result;
}

// satisfies -Wmissing-declarations
int llama_cli(int argc, char ** argv);

int llama_cli(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (params.server_base.empty() && is_diffusion_model(params.model.path)) {
        return llama_cli_diffusion(params);
    }

    cli_context ctx_cli(params);

    if (!ctx_cli.init()) {
        return 1;
    }

    return ctx_cli.run();
}
