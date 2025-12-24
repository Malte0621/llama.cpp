#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
# include <winsock2.h>
typedef SOCKET sockfd_t;
#else
# include <sys/types.h>
typedef int sockfd_t;
#endif

// cross-platform socket descriptor
#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

struct socket_t {
    sockfd_t fd;
    // scheme of transport ("tcp", "rdma", ...)
    std::string scheme;
    // opaque transport-specific context (e.g., rdma_cm_id*)
    void * transport_ctx = nullptr;

    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t();
};

// Transport selection helper
inline bool parse_endpoint_scheme(const std::string & endpoint, std::string & scheme, std::string & host, int & port) {
    // Supported: [scheme://]host:port
    std::string work = endpoint;
    // default scheme is TCP
    scheme = "tcp";
    size_t pos = work.find("://");
    if (pos != std::string::npos) {
        scheme = work.substr(0, pos);
        work = work.substr(pos + 3);
    }
    size_t p = work.find(':');
    if (p == std::string::npos) return false;
    host = work.substr(0, p);
    port = std::stoi(work.substr(p+1));
    return true;
}

// Server-side helpers
std::shared_ptr<socket_t> transport_create_server(const std::string & scheme, const char * host, int port);
std::shared_ptr<socket_t> transport_accept(std::shared_ptr<socket_t> srv);

// Client-side helpers
std::shared_ptr<socket_t> transport_connect(const std::string & scheme, const char * host, int port);

// Message helpers
bool transport_send_msg(std::shared_ptr<socket_t> sock, const void * msg, size_t msg_size);
bool transport_recv_msg(std::shared_ptr<socket_t> sock, void * msg, size_t msg_size);
bool transport_recv_vector(std::shared_ptr<socket_t> sock, std::vector<uint8_t> & input);

// Raw data
bool transport_send_data(std::shared_ptr<socket_t> sock, const void * data, size_t size);
bool transport_recv_data(std::shared_ptr<socket_t> sock, void * data, size_t size);

// runtime-configured helpers (can be inlined for tests)
inline uint64_t rpc_get_max_msg_size() {
    const char * v = std::getenv("GGML_RPC_MAX_MSG_SIZE");
    if (!v) return 256ull * 1024ull * 1024ull;
    try { return std::stoull(v); } catch (...) { return 256ull * 1024ull * 1024ull; }
}
inline int rpc_get_timeout_secs() {
    const char * v = std::getenv("GGML_RPC_TIMEOUT");
    if (!v) return 30;
    try { return std::stoi(v); } catch (...) { return 30; }
}
inline int rpc_get_max_clients() {
    const char * v = std::getenv("GGML_RPC_MAX_CLIENTS");
    if (!v) return 32;
    try { return std::stoi(v); } catch (...) { return 32; }
}

// Utility to set timeouts on a socket (internal)
bool set_socket_timeouts(sockfd_t sockfd);
