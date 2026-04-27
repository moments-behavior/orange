#include "server_endpoints.h"
#include "json.hpp"
#include <fstream>
#include <stdexcept>

std::vector<ServerEndpoint> load_server_endpoints(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("endpoints file not found: " + path);
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception &e) {
        throw std::runtime_error("endpoints JSON parse error in " + path +
                                 ": " + e.what());
    }

    const int default_port = j.value("default_port", 0);

    if (!j.contains("servers") || !j["servers"].is_array()) {
        throw std::runtime_error("endpoints missing 'servers' array: " + path);
    }

    std::vector<ServerEndpoint> out;
    for (const auto &s : j["servers"]) {
        ServerEndpoint e;
        e.name = s.at("name").get<std::string>();
        e.host = s.at("host").get<std::string>();
        e.port = s.value("port", default_port);
        if (e.port <= 0) {
            throw std::runtime_error("server '" + e.name +
                                     "' has no port and no default_port set");
        }
        out.push_back(std::move(e));
    }

    if (out.empty()) {
        throw std::runtime_error("endpoints 'servers' array is empty: " + path);
    }
    return out;
}
