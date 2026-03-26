#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

// Minimal single-threaded MJPEG HTTP server.
// One instance per camera. Connect with a browser or:
//   ffplay http://HOST:PORT
class MjpegServer {
public:
    explicit MjpegServer(int port)
        : port_(port), running_(false), server_fd_(-1) {}

    ~MjpegServer() { if (running_) stop(); }

    void start() {
        running_ = true;
        thread_ = std::thread([this] { serve(); });
        printf("MJPEG stream on port %d\n", port_);
        fflush(stdout);
    }

    void stop() {
        running_ = false;
        if (server_fd_ >= 0) { ::close(server_fd_); server_fd_ = -1; }
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> lk(mu_);
        for (int fd : clients_) ::close(fd);
        clients_.clear();
    }

    // Push a JPEG buffer to all connected viewers. Thread-safe.
    void push(const std::vector<uint8_t>& jpeg) {
        if (jpeg.empty()) return;
        const std::string hdr =
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<int> dead;
        for (int fd : clients_) {
            if (::send(fd, hdr.data(), hdr.size(), MSG_NOSIGNAL) < 0 ||
                ::send(fd, jpeg.data(), jpeg.size(), MSG_NOSIGNAL) < 0)
                dead.push_back(fd);
        }
        for (int fd : dead) {
            ::close(fd);
            clients_.erase(std::remove(clients_.begin(), clients_.end(), fd),
                           clients_.end());
        }
    }

    int port() const { return port_; }

private:
    int port_;
    std::atomic<bool> running_;
    int server_fd_;
    std::thread thread_;
    std::mutex mu_;
    std::vector<int> clients_;

    void serve() {
        server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) return;
        int opt = 1;
        ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) < 0) {
            printf("MJPEG bind failed on port %d\n", port_);
            return;
        }
        ::listen(server_fd_, 5);

        while (running_) {
            fd_set fds; FD_ZERO(&fds); FD_SET(server_fd_, &fds);
            timeval tv{0, 100000}; // 100 ms poll
            if (::select(server_fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0)
                continue;
            int cli = ::accept(server_fd_, nullptr, nullptr);
            if (cli < 0) continue;
            // Drain the HTTP request (we don't care about path/headers)
            char buf[2048];
            ::recv(cli, buf, sizeof(buf) - 1, 0);
            // Send MJPEG response
            const std::string resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n"
                "Connection: keep-alive\r\n"
                "Cache-Control: no-cache\r\n\r\n";
            ::send(cli, resp.data(), resp.size(), MSG_NOSIGNAL);
            std::lock_guard<std::mutex> lk(mu_);
            clients_.push_back(cli);
        }
    }
};
