#include "rpc-transport.h"
#include "ggml-rpc.h"

#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cinttypes>

#ifndef _WIN32
# include <unistd.h>
#endif

// socket_t destructor implementation (defined in header)
socket_t::~socket_t() {
#ifdef _WIN32
    closesocket(this->fd);
#else
    close(this->fd);
#endif
}

// For TCP implementation we reuse the existing helpers from ggml-rpc.cpp
// by declaring them here as extern "C"-like symbols; they are static in
// ggml-rpc.cpp so we reimplement small wrappers using the socket APIs.

#ifndef _WIN32
# include <arpa/inet.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <netdb.h>
# include <sys/uio.h>
# include <unistd.h>
#else
# include <winsock2.h>
# include <ws2tcpip.h>
#endif

#include <memory>
#include <iostream>
#if defined(__has_include)
#  if __has_include(<zstd.h>)
#    include <zstd.h>
#    define GGML_RPC_HAVE_ZSTD 1
#  else
#    define GGML_RPC_HAVE_ZSTD 0
#  endif
#else
#  define GGML_RPC_HAVE_ZSTD 0
#endif
#include <vector>

// socket_t is defined in rpc-transport.h

// runtime-configurable limits
static constexpr uint64_t RPC_DEFAULT_MAX_MSG_SIZE = 256ull * 1024ull * 1024ull; // 256 MiB
static constexpr int RPC_DEFAULT_TIMEOUT_SECS = 30; // seconds
static constexpr int RPC_DEFAULT_MAX_CLIENTS = 32;

// rpc_get_* helpers are implemented inline in the header for cross-TU visibility


bool set_socket_timeouts(sockfd_t sockfd) {
    int secs = rpc_get_timeout_secs();
#ifdef _WIN32
    DWORD tv = (DWORD)(secs * 1000);
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv)) != 0) return false;
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv)) != 0) return false;
#else
    struct timeval tv;
    tv.tv_sec = secs;
    tv.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) return false;
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) return false;
#endif
    return true;
}

// Compression helpers
#if GGML_RPC_HAVE_ZSTD
// Compress `src` into `out` using zstd (fast mode). Return true on success.
bool rpc_compress_buffer(const void * src, size_t src_size, std::vector<uint8_t> & out) {
    if (!src || src_size == 0) {
        out.clear();
        return true;
    }
    size_t bound = ZSTD_compressBound(src_size);
    out.resize(bound);
    // Use fast compression level for low CPU/low latency
    int level = 1; // fast
    size_t csize = ZSTD_compress(out.data(), bound, src, src_size, level);
    if (ZSTD_isError(csize)) return false;
    out.resize(csize);
    return true;
}

// Decompress `src` of size `src_size` into `dst` of size `dst_size`.
bool rpc_decompress_buffer(const void * src, size_t src_size, void * dst, size_t dst_size) {
    if (!src) return false;
    size_t res = ZSTD_decompress(dst, dst_size, src, src_size);
    if (ZSTD_isError(res)) return false;
    if (res != dst_size) return false; // must match expected size
    return true;
}
#else
// Stubs when zstd is not available
bool rpc_compress_buffer(const void * src, size_t src_size, std::vector<uint8_t> & out) {
    (void)src; (void)src_size; out.clear(); return false;
}
bool rpc_decompress_buffer(const void * src, size_t src_size, void * dst, size_t dst_size) {
    (void)src; (void)src_size; (void)dst; (void)dst_size; return false;
}
#endif


static bool tcp_set_no_delay(sockfd_t sockfd) {
    if (!rpc_get_no_delay()) return true; // explicitly disabled
    int flag = 1;
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool tcp_set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

// [AI] Added socket buffer tuning (GGML_RPC_SNDBUF / GGML_RPC_RCVBUF)
static bool tcp_set_bufs(sockfd_t sockfd) {
    int snd = rpc_get_send_buf_size();
    int rcv = rpc_get_recv_buf_size();
    bool ok = true;
    if (snd > 0) {
        if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char *)&snd, sizeof(int)) != 0) ok = false;
    }
    if (rcv > 0) {
        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *)&rcv, sizeof(int)) != 0) ok = false;
    }
    return ok;
}

static void tcp_set_reuse_port_if_requested(sockfd_t sockfd) {
#ifdef SO_REUSEPORT
    if (rpc_get_enable_reuseport()) {
        int flag = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, (char *)&flag, sizeof(int));
    }
#endif
}

static void tcp_set_quickack(sockfd_t sockfd) {
#ifdef TCP_QUICKACK
    if (rpc_get_quickack()) {
        int flag = 1;
        setsockopt(sockfd, IPPROTO_TCP, TCP_QUICKACK, (char *)&flag, sizeof(int));
    }
#endif
}

static std::shared_ptr<socket_t> tcp_socket_connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return nullptr;
    auto sock_ptr = std::make_shared<socket_t>(sockfd);
    sock_ptr->scheme = "tcp";
    sock_ptr->transport_ctx = nullptr;
    if (!tcp_set_no_delay(sockfd)) return nullptr;

    // apply buffer tunings for the socket if requested
    tcp_set_bufs(sockfd);

    // Resolve host via getaddrinfo (reentrant and protocol-agnostic)
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    if (getaddrinfo(host, portstr, &hints, &res) != 0) {
        return nullptr;
    }

    bool connected = false;
    for (struct addrinfo *p = res; p != NULL; p = p->ai_next) {
        if (connect(sock_ptr->fd, p->ai_addr, (int)p->ai_addrlen) == 0) {
            connected = true;
            break;
        }
    }
    freeaddrinfo(res);
    if (!connected) return nullptr;

    // apply timeouts and keepalive after connection
    set_socket_timeouts(sock_ptr->fd);
    int ka = 1;
    setsockopt(sock_ptr->fd, SOL_SOCKET, SO_KEEPALIVE, (char *)&ka, sizeof(ka));
    tcp_set_quickack(sock_ptr->fd);
    return sock_ptr;
}

static std::shared_ptr<socket_t> tcp_socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    if (client_socket_fd < 0) return nullptr;
    auto client_socket = std::make_shared<socket_t>(client_socket_fd);
    client_socket->scheme = "tcp";
    client_socket->transport_ctx = nullptr;
    if (!tcp_set_no_delay(client_socket_fd)) return nullptr;
    // tune buffers and quick ack if requested
    tcp_set_bufs(client_socket_fd);
    set_socket_timeouts(client_socket_fd);
    int ka = 1;
    setsockopt(client_socket_fd, SOL_SOCKET, SO_KEEPALIVE, (char *)&ka, sizeof(ka));
    tcp_set_quickack(client_socket_fd);
    return client_socket;
}

static std::shared_ptr<socket_t> tcp_create_server(const char * host, int port) {
    // Use getaddrinfo for IPv4/IPv6 and hostname support
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    if (getaddrinfo(host, portstr, &hints, &res) != 0) {
        return nullptr;
    }

    // helper to close a socket on the current platform
    auto close_socket = [](int fd) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
    };

    std::shared_ptr<socket_t> result = nullptr;
    for (struct addrinfo *p = res; p != NULL; p = p->ai_next) {
        int sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (sockfd < 0) continue;
        // allow address reuse
        if (!tcp_set_reuse_addr(sockfd)) {
            close_socket(sockfd);
            continue;
        }
        // optionally enable reuseport for better scaling on Linux
        tcp_set_reuse_port_if_requested(sockfd);
        // tune buffers if requested
        tcp_set_bufs(sockfd);
        if (bind(sockfd, p->ai_addr, (int)p->ai_addrlen) < 0) {
            close_socket(sockfd);
            continue;
        }
        if (listen(sockfd, SOMAXCONN) < 0) {
            close_socket(sockfd);
            continue;
        }
        set_socket_timeouts(sockfd);
        result = std::make_shared<socket_t>(sockfd);
        result->scheme = "tcp";
        result->transport_ctx = nullptr;
        break;
    }
    freeaddrinfo(res);
    return result;
}

static bool tcp_send_data(std::shared_ptr<socket_t> sock, const void * data, size_t size) {
    if (!sock) return false;
#ifdef _WIN32
    if (sock->fd == INVALID_SOCKET) return false;
#else
    if (sock->fd < 0) return false;
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
#ifdef MSG_NOSIGNAL
        int flags = MSG_NOSIGNAL;
#else
        int flags = 0;
#endif
        ssize_t n = send(sock->fd, (const char *)data + bytes_sent, (int)(size - bytes_sent), flags);
        if (n < 0) {
#ifdef _WIN32
            return false;
#else
            if (errno == EINTR) continue; // retry on signal
            return false;
#endif
        }
        bytes_sent += (size_t)n;
    }
    return true;
} 

static bool tcp_recv_data(std::shared_ptr<socket_t> sock, void * data, size_t size) {
    if (!sock) return false;
#ifdef _WIN32
    if (sock->fd == INVALID_SOCKET) return false;
#else
    if (sock->fd < 0) return false;
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sock->fd, (char *)data + bytes_recv, (int)(size - bytes_recv), 0);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EINTR) continue; // retry on signal
            fprintf(stderr, "tcp_recv_data: recv failed (bytes_recv=%zu, size_to_recv=%zu): %s\n", bytes_recv, size - bytes_recv, strerror(errno));
#else
            fprintf(stderr, "tcp_recv_data: recv failed (bytes_recv=%zu, size_to_recv=%zu)\n", bytes_recv, size - bytes_recv);
#endif
            return false;
        }
        if (n == 0) {
            fprintf(stderr, "tcp_recv_data: peer closed connection\n");
            return false; // peer closed
        }
        bytes_recv += (size_t)n;
    }
    return true;
} 

// Attempt to read and discard `size` bytes from the socket to keep the stream aligned.
static bool tcp_drain_data(std::shared_ptr<socket_t> sock, size_t size) {
    if (!sock) return false;
    const size_t chunk = 64 * 1024;
    std::vector<uint8_t> tmp;
    try { tmp.resize((size_t)(std::min)(chunk, size)); } catch (...) { return false; }
    size_t drained = 0;
    while (drained < size) {
        size_t to_read = (std::min)(chunk, size - drained);
        if (!tcp_recv_data(sock, tmp.data(), to_read)) return false;
        drained += to_read;
    }
    return true;
}

std::shared_ptr<socket_t> transport_create_server(const std::string & scheme, const char * host, int port) {
    if (scheme == "rdma") {
#ifdef GGML_RPC_RDMA
        // Minimal RDMA server setup: create event channel and listener
        struct rdma_server_ctx {
            struct rdma_event_channel * ec;
            struct rdma_cm_id * listener;
        };

        struct rdma_server_ctx * ctx = new rdma_server_ctx();
        ctx->ec = rdma_create_event_channel();
        if (!ctx->ec) {
            fprintf(stderr, "rdma_create_event_channel failed\n");
            delete ctx;
            return nullptr;
        }
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        char portstr[16];
        snprintf(portstr, sizeof(portstr), "%d", port);
        if (getaddrinfo(host, portstr, &hints, &res) != 0) {
            fprintf(stderr, "getaddrinfo failed for %s:%d\n", host, port);
            rdma_destroy_event_channel(ctx->ec);
            delete ctx;
            return nullptr;
        }
        if (rdma_create_id(ctx->ec, &ctx->listener, NULL, RDMA_PS_TCP) != 0) {
            fprintf(stderr, "rdma_create_id failed\n");
            freeaddrinfo(res);
            rdma_destroy_event_channel(ctx->ec);
            delete ctx;
            return nullptr;
        }
        if (rdma_bind_addr(ctx->listener, res->ai_addr) != 0) {
            fprintf(stderr, "rdma_bind_addr failed\n");
            rdma_destroy_id(ctx->listener);
            freeaddrinfo(res);
            rdma_destroy_event_channel(ctx->ec);
            delete ctx;
            return nullptr;
        }
        freeaddrinfo(res);
        if (rdma_listen(ctx->listener, 10) != 0) {
            fprintf(stderr, "rdma_listen failed\n");
            rdma_destroy_id(ctx->listener);
            rdma_destroy_event_channel(ctx->ec);
            delete ctx;
            return nullptr;
        }
        // Create a placeholder socket_t to carry the RDMA listen context
        auto s = std::make_shared<socket_t>(-1);
        s->scheme = "rdma";
        s->transport_ctx = ctx;
        return s;
#else
        fprintf(stderr, "RDMA transport not enabled in build (GGML_RPC_RDMA=OFF)\n");
        return nullptr;
#endif
    }
    // default to tcp
    return tcp_create_server(host, port);
}

std::shared_ptr<socket_t> transport_accept(std::shared_ptr<socket_t> srv) {
    if (srv->scheme == "rdma") {
#ifdef GGML_RPC_RDMA
        struct rdma_server_ctx { struct rdma_event_channel * ec; struct rdma_cm_id * listener; };
        rdma_server_ctx * ctx = (rdma_server_ctx *)srv->transport_ctx;
        if (!ctx || !ctx->ec) return nullptr;
        struct rdma_cm_event * event = NULL;
        if (rdma_get_cm_event(ctx->ec, &event)) {
            fprintf(stderr, "rdma_get_cm_event failed\n");
            return nullptr;
        }
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            rdma_ack_cm_event(event);
            fprintf(stderr, "unexpected RDMA event: %d\n", event->event);
            return nullptr;
        }
        struct rdma_cm_id * client_id = event->id;
        rdma_ack_cm_event(event);
        // Accept the connection (we don't configure real buffer/qp here yet)
        struct rdma_conn_param conn_param;
        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.initiator_depth = conn_param.responder_resources = 1;
        if (rdma_accept(client_id, &conn_param) != 0) {
            fprintf(stderr, "rdma_accept failed\n");
            rdma_destroy_id(client_id);
            return nullptr;
        }
        // Create socket wrapper and attach client_id
        auto s = std::make_shared<socket_t>(-1);
        s->scheme = "rdma";
        s->transport_ctx = client_id;
        return s;
#else
        fprintf(stderr, "RDMA transport not enabled in build (GGML_RPC_RDMA=OFF)\n");
        return nullptr;
#endif
    }
    // only TCP implemented currently
    return tcp_socket_accept(srv->fd);
}

std::shared_ptr<socket_t> transport_connect(const std::string & scheme, const char * host, int port) {
    if (scheme == "rdma") {
#ifdef GGML_RPC_RDMA
        struct rdma_conn_ctx { struct rdma_cm_id * id; };
        struct rdma_conn_ctx * ctx = new rdma_conn_ctx();
        ctx->id = NULL;
        struct rdma_event_channel * ec = rdma_create_event_channel();
        if (!ec) {
            fprintf(stderr, "rdma_create_event_channel failed\n");
            delete ctx;
            return nullptr;
        }
        if (rdma_create_id(ec, &ctx->id, NULL, RDMA_PS_TCP) != 0) {
            fprintf(stderr, "rdma_create_id failed\n");
            rdma_destroy_event_channel(ec);
            delete ctx;
            return nullptr;
        }
        struct addrinfo hints{}, *res = NULL;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        char portstr[16];
        snprintf(portstr, sizeof(portstr), "%d", port);
        if (getaddrinfo(host, portstr, &hints, &res) != 0) {
            fprintf(stderr, "getaddrinfo failed for %s:%d\n", host, port);
            rdma_destroy_id(ctx->id);
            rdma_destroy_event_channel(ec);
            delete ctx;
            return nullptr;
        }
        if (rdma_resolve_addr(ctx->id, NULL, res->ai_addr, 2000) != 0) {
            fprintf(stderr, "rdma_resolve_addr failed\n");
            freeaddrinfo(res);
            rdma_destroy_id(ctx->id);
            rdma_destroy_event_channel(ec);
            delete ctx;
            return nullptr;
        }
        // Note: we skip waiting for route resolved events for brevity
        freeaddrinfo(res);
        // Connect (we don't wait for established event here)
        struct rdma_conn_param conn_param;
        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.initiator_depth = conn_param.responder_resources = 1;
        if (rdma_connect(ctx->id, &conn_param) != 0) {
            fprintf(stderr, "rdma_connect failed\n");
            rdma_destroy_id(ctx->id);
            rdma_destroy_event_channel(ec);
            delete ctx;
            return nullptr;
        }
        auto s = std::make_shared<socket_t>(-1);
        s->scheme = "rdma";
        s->transport_ctx = ctx->id;
        return s;
#else
        fprintf(stderr, "RDMA transport not enabled in build (GGML_RPC_RDMA=OFF)\n");
        return nullptr;
#endif
    }
    return tcp_socket_connect(host, port);
}

// Wire format: [uint8_t comp_flag][uint64_t orig_size][uint64_t payload_size][payload bytes...]
#pragma pack(push, 1)
struct rpc_wire_header_packed {
    uint8_t  comp;        // 0 = none, 1 = zstd
    uint64_t orig_size;   // original uncompressed size
    uint64_t payload_size; // size of payload following the header
};
#pragma pack(pop)

bool transport_send_msg(std::shared_ptr<socket_t> sock, const void * msg, size_t msg_size) {
    if (!sock) return false;

    uint8_t comp_flag = 0;
    const void * payload_ptr = msg;
    size_t payload_size = msg_size;
    std::vector<uint8_t> compressed_buf;

    // Attempt compression if enabled and beneficial
    if (msg_size > 0 && rpc_compression_enabled() && msg_size >= rpc_get_compression_min_size()) {
        if (rpc_get_compression() == "zstd") {
#if GGML_RPC_HAVE_ZSTD
            if (rpc_compress_buffer(msg, msg_size, compressed_buf)) {
                if (compressed_buf.size() < msg_size) {
                    comp_flag = 1;
                    payload_ptr = compressed_buf.data();
                    payload_size = compressed_buf.size();
                }
            }
#endif
        }
    }

    rpc_wire_header_packed hdr{};
    hdr.comp = comp_flag;
    hdr.orig_size = (uint64_t)msg_size;
    hdr.payload_size = (uint64_t)payload_size;

#if defined(__unix__) || defined(__APPLE__)
    // [AI] Fix: Use the reliable tcp_send_data helper which loops until the full buffer is written.
    // Using sendmsg without handling partial writes could leave the stream in a partially-sent
    // state which later leads to "malformed" messages on the receiver.
    if (sock->fd < 0) return false;
    if (!tcp_send_data(sock, &hdr, sizeof(hdr))) return false;
    if (payload_size == 0) return true;
    return tcp_send_data(sock, payload_ptr, payload_size);
#else
#ifdef _WIN32
    // On Windows use the reliable send helper as well. WSASend may succeed but send fewer bytes
    // than requested - check sent bytes and fall back to looped sending.
    if (sock->fd == INVALID_SOCKET) return false;
    // Send header
    if (!tcp_send_data(sock, &hdr, sizeof(hdr))) return false;
    if (payload_size == 0) return true;
    return tcp_send_data(sock, payload_ptr, payload_size);
#else
    // Fallback: send header then data
    if (!tcp_send_data(sock, &hdr, sizeof(hdr))) return false;
    if (payload_size == 0) return true;
    return tcp_send_data(sock, payload_ptr, payload_size);
#endif
#endif
} 

bool transport_recv_msg(std::shared_ptr<socket_t> sock, void * msg, size_t msg_size) {
    if (!sock) return false;
#ifdef _WIN32
    if (sock->fd == INVALID_SOCKET) return false;
#else
    if (sock->fd < 0) return false;
#endif
    // Read header
    rpc_wire_header_packed hdr;
    if (!tcp_recv_data(sock, &hdr, sizeof(hdr))) return false;
    uint64_t max_size = rpc_get_max_msg_size();
    if (hdr.orig_size > max_size) {
        fprintf(stderr, "Rejected message larger than max allowed (%" PRIu64 " > %" PRIu64 ")\n", hdr.orig_size, max_size);
        // drain payload if present to keep stream aligned
        if (hdr.payload_size > 0) {
            tcp_drain_data(sock, (size_t)hdr.payload_size);
        }
        return false;
    }
    if (hdr.orig_size != msg_size) {
        fprintf(stderr, "Mismatched expected message size (%zu) != hdr.orig_size (%" PRIu64 ")\n", msg_size, hdr.orig_size);
        // drain payload to keep stream aligned
        if (hdr.payload_size > 0) {
            tcp_drain_data(sock, (size_t)hdr.payload_size);
        }
        return false;
    }
    if (hdr.payload_size == 0) return true;
    // read payload
    std::vector<uint8_t> tmp;
    if (hdr.comp == 0) {
        // uncompressed: read directly into msg
        if (hdr.payload_size > msg_size) {
            fprintf(stderr, "Payload size (%" PRIu64 ") exceeds expected buffer size (%zu)\n", hdr.payload_size, msg_size);
            tcp_drain_data(sock, (size_t)hdr.payload_size);
            return false;
        }
        return tcp_recv_data(sock, msg, (size_t)hdr.payload_size);
    } else if (hdr.comp == 1) {
        // compressed: read into tmp and decompress
        try { tmp.resize((size_t)hdr.payload_size); } catch (...) { return false; }
        if (!tcp_recv_data(sock, tmp.data(), tmp.size())) return false;
        // decompress into msg buffer (msg_size bytes)
        if (!rpc_decompress_buffer(tmp.data(), tmp.size(), msg, msg_size)) return false;
        return true;
    }
    return false;
}

bool transport_recv_vector(std::shared_ptr<socket_t> sock, std::vector<uint8_t> & input) {
    if (!sock) return false;
#ifdef _WIN32
    if (sock->fd == INVALID_SOCKET) return false;
#else
    if (sock->fd < 0) return false;
#endif
    rpc_wire_header_packed hdr;
    if (!tcp_recv_data(sock, &hdr, sizeof(hdr))) return false;
    uint64_t max_size = rpc_get_max_msg_size();
    if (hdr.orig_size > max_size) {
        fprintf(stderr, "Rejected vector message larger than max allowed (%" PRIu64 " > %" PRIu64 ")\n", hdr.orig_size, max_size);
        if (hdr.payload_size > 0) tcp_drain_data(sock, (size_t)hdr.payload_size);
        return false;
    }
    try {
        input.resize((size_t)hdr.orig_size);
    } catch (const std::bad_alloc & e) {
        fprintf(stderr, "Failed to allocate input buffer of size %" PRIu64 "\n", hdr.orig_size);
        if (hdr.payload_size > 0) tcp_drain_data(sock, (size_t)hdr.payload_size);
        return false;
    }
    if (hdr.payload_size == 0) return true;
    if (hdr.comp == 0) {
        if (hdr.payload_size > hdr.orig_size) {
            fprintf(stderr, "Uncompressed payload_size (%" PRIu64 ") > orig_size (%" PRIu64 ")\n", hdr.payload_size, hdr.orig_size);
            tcp_drain_data(sock, (size_t)hdr.payload_size);
            return false;
        }
        return tcp_recv_data(sock, input.data(), (size_t)hdr.payload_size);
    } else if (hdr.comp == 1) {
        std::vector<uint8_t> tmp;
        try { tmp.resize((size_t)hdr.payload_size); } catch (...) { return false; }
        if (!tcp_recv_data(sock, tmp.data(), tmp.size())) return false;
        if (!rpc_decompress_buffer(tmp.data(), tmp.size(), input.data(), hdr.orig_size)) return false;
        return true;
    }
    return false;
}

// Raw data
bool transport_send_data(std::shared_ptr<socket_t> sock, const void * data, size_t size) {
    return tcp_send_data(sock, data, size);
}

bool transport_recv_data(std::shared_ptr<socket_t> sock, void * data, size_t size) {
    return tcp_recv_data(sock, data, size);
}

