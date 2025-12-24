#include "rpc-transport.h"
#include <cassert>
#include <iostream>
#include <cstdlib>

int main() {
    // default
#ifdef _WIN32
    _putenv_s("GGML_RPC_MAX_MSG_SIZE", "");
#else
    unsetenv("GGML_RPC_MAX_MSG_SIZE");
#endif
    uint64_t default_size = rpc_get_max_msg_size();
    assert(default_size == 256ull * 1024ull * 1024ull);

    // set env var
#ifdef _WIN32
    _putenv_s("GGML_RPC_MAX_MSG_SIZE", "1048576");
#else
    setenv("GGML_RPC_MAX_MSG_SIZE", "1048576", 1);
#endif
    uint64_t new_size = rpc_get_max_msg_size();
    assert(new_size == 1048576ull);

    // max clients
#ifdef _WIN32
    _putenv_s("GGML_RPC_MAX_CLIENTS", "");
#else
    unsetenv("GGML_RPC_MAX_CLIENTS");
#endif
    int default_clients = rpc_get_max_clients();
    assert(default_clients == 32);

#ifdef _WIN32
    _putenv_s("GGML_RPC_MAX_CLIENTS", "8");
#else
    setenv("GGML_RPC_MAX_CLIENTS", "8", 1);
#endif
    assert(rpc_get_max_clients() == 8);

    std::cout << "rpc-transport config tests passed\n";
    return 0;
}
