#pragma once
#include <string>
#include <vector>

struct ServerEndpoint {
    std::string name;
    std::string host;
    int port;
};

std::vector<ServerEndpoint> load_server_endpoints(const std::string &path);
