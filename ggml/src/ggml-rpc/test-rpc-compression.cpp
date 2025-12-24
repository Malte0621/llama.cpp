#include "rpc-transport.h"
#include <iostream>
#include <cassert>
#include <string>

int main() {
#if GGML_RPC_HAVE_ZSTD
    std::string s;
    for (int i = 0; i < 4096; ++i) s.push_back('A' + (i%26));
    std::vector<uint8_t> out;
    bool ok = rpc_compress_buffer(s.data(), s.size(), out);
    if (!ok) { std::cerr << "Compression failed\n"; return 2; }
    std::vector<uint8_t> dec(s.size());
    ok = rpc_decompress_buffer(out.data(), out.size(), dec.data(), dec.size());
    if (!ok) { std::cerr << "Decompression failed\n"; return 3; }
    if (std::string((char*)dec.data(), dec.size()) != s) { std::cerr << "Roundtrip mismatch\n"; return 4; }
    std::cout << "rpc-compression test passed (zstd)\n";
#else
    std::cout << "zstd not available; test skipped\n";
#endif
    return 0;
}
