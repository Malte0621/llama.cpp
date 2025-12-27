#include "rpc-transport.h"
#include "../../include/ggml-rpc.h"
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <chrono>

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

int main() {
    // Minimal test server: accept one connection and respond to HELLO and DEVICE_COUNT
#if defined(_WIN32)
    // initialize winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed; skipping test\n";
        return 0;
    }
#endif

    // create listening socket
#if defined(_WIN32)
    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd == INVALID_SOCKET) {
        std::cerr << "socket() failed\n";
        return 0;
    }
#else
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::cerr << "socket() failed\n";
        return 0;
    }
#endif

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(50505);

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        std::cerr << "bind() failed\n";
        return 0;
    }
    if (listen(listen_fd, 1) != 0) {
        std::cerr << "listen() failed\n";
        return 0;
    }

    std::atomic<bool> stop_flag{false};
    std::thread server_thread([&]() {
        // accept one client and serve it
        struct sockaddr_in cli_addr{};
        socklen_t cli_len = sizeof(cli_addr);
#if defined(_WIN32)
        SOCKET cli = accept(listen_fd, (struct sockaddr*)&cli_addr, &cli_len);
        if (cli == INVALID_SOCKET) return;
#else
        int cli = accept(listen_fd, (struct sockaddr*)&cli_addr, &cli_len);
        if (cli < 0) return;
#endif

        // local copy of header structure
        #pragma pack(push, 1)
        struct rpc_wire_header_packed { uint8_t comp; uint64_t orig_size; uint64_t payload_size; } hdr;
        #pragma pack(pop)

        const uint8_t CMD_HELLO = 14;
        const uint8_t CMD_DEVICE_COUNT = 1;

        while (!stop_flag.load()) {
            uint8_t cmd;
#if defined(_WIN32)
            int n = recv(cli, (char *)&cmd, 1, 0);
#else
            ssize_t n = recv(cli, (char *)&cmd, 1, 0);
#endif
            if (n <= 0) break;
            // read header
#if defined(_WIN32)
            int h = recv(cli, (char *)&hdr, sizeof(hdr), 0);
#else
            ssize_t h = recv(cli, (char *)&hdr, sizeof(hdr), 0);
#endif
            if (h != (int)sizeof(hdr)) break;

            // read payload (if any) and discard
            if (hdr.payload_size > 0) {
                std::vector<uint8_t> tmp((size_t)hdr.payload_size);
#if defined(_WIN32)
                int r = recv(cli, (char *)tmp.data(), (int)tmp.size(), 0);
#else
                ssize_t r = recv(cli, (char *)tmp.data(), (int)tmp.size(), 0);
#endif
                if (r != (int)tmp.size()) break;
            }

            if (cmd == CMD_HELLO) {
                // send version response (3 bytes)
                rpc_wire_header_packed ohdr{0, 3, 3};
                uint8_t ver[3] = { (uint8_t)RPC_PROTO_MAJOR_VERSION, (uint8_t)RPC_PROTO_MINOR_VERSION, (uint8_t)RPC_PROTO_PATCH_VERSION };
#if defined(_WIN32)
                send(cli, (const char *)&ohdr, sizeof(ohdr), 0);
                send(cli, (const char *)ver, sizeof(ver), 0);
#else
                send(cli, &ohdr, sizeof(ohdr), 0);
                send(cli, ver, sizeof(ver), 0);
#endif
            } else if (cmd == CMD_DEVICE_COUNT) {
                // send device count (uint32_t)
                uint32_t dev_count = 1;
                rpc_wire_header_packed ohdr{0, sizeof(dev_count), sizeof(dev_count)};
#if defined(_WIN32)
                send(cli, (const char *)&ohdr, sizeof(ohdr), 0);
                send(cli, (const char *)&dev_count, sizeof(dev_count), 0);
#else
                send(cli, &ohdr, sizeof(ohdr), 0);
                send(cli, &dev_count, sizeof(dev_count), 0);
#endif
            } else {
                break;
            }
        }
#if defined(_WIN32)
        closesocket(cli);
#else
        close(cli);
#endif
    });
    // allow some time for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const char * endpoint = "127.0.0.1:50505";
    const int n_threads = 8;
    const int n_iters = 500;

    std::atomic<int> ok_count{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < n_iters; ++i) {
                uint32_t count = 0;
                if (ggml_rpc_send_device_count(endpoint, &count)) {
                    ok_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // delay a bit to avoid tight spinning on failures
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }

    for (auto & th : threads) th.join();

    stop_flag.store(true);
    // close the listening socket
#if defined(_WIN32)
    closesocket(listen_fd);
    WSACleanup();
#else
    close(listen_fd);
#endif
    server_thread.join();

    if (ok_count == n_threads * n_iters) {
        std::cout << "rpc-concurrency-test passed\n";
        return 0;
    } else {
        std::cerr << "rpc-concurrency-test failed: ok_count=" << ok_count << " expected=" << (n_threads * n_iters) << std::endl;
        return 1;
    }
}
