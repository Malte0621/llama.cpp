#include "rpc-transport.h"
#include <cassert>
#include <iostream>

int main() {
    std::string scheme, host;
    int port;
    bool ok;

    ok = parse_endpoint_scheme("127.0.0.1:50052", scheme, host, port);
    assert(ok && scheme == "tcp" && host == "127.0.0.1" && port == 50052);

    ok = parse_endpoint_scheme("tcp://127.0.0.1:50052", scheme, host, port);
    assert(ok && scheme == "tcp" && host == "127.0.0.1" && port == 50052);

    ok = parse_endpoint_scheme("rdma://10.0.0.1:1234", scheme, host, port);
    assert(ok && scheme == "rdma" && host == "10.0.0.1" && port == 1234);

    std::cout << "rpc-transport tests passed\n";
    return 0;
}
