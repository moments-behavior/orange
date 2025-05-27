#include "serial.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <sstream>

SerialPort::SerialPort() : fd_(-1) {}
SerialPort::~SerialPort() { close(); }

bool SerialPort::open(const std::string& port_name, int baud_rate) {
    fd_ = ::open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;

    struct termios tty {};
    if (tcgetattr(fd_, &tty) != 0) return false;

    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag = tty.c_oflag = tty.c_lflag = 0;
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) return false;

    return true;
}

void SerialPort::close() {
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SerialPort::is_open() const {
    return fd_ != -1;
}

bool SerialPort::write(const std::string& data) {
    if (!is_open()) return false;
    return ::write(fd_, data.c_str(), data.size()) > 0;
}

std::string SerialPort::read() {
    if (!is_open()) return "";
    char buf[256];
    int n = ::read(fd_, buf, sizeof(buf));
    if (n > 0) {
        return std::string(buf, n);
    }
    return ""; 
}

std::vector<std::string> SerialPort::list_available_ports() {
    std::vector<std::string> ports;
    DIR* dev_dir = opendir("/dev");
    if (!dev_dir) return ports;

    struct dirent* entry;
    while ((entry = readdir(dev_dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name.find("ttyUSB") != std::string::npos || name.find("ttyACM") != std::string::npos) {
            ports.push_back("/dev/" + name);
        }
    }
    closedir(dev_dir);
    return ports;
}

void SerialPort::send_pump_command(char pump, bool push, int cycles, int delay_us) {
    if (!is_open()) return;

    std::ostringstream oss;
    oss << (push ? 'h' : 'l') << pump << " " << cycles << " " << delay_us << "\n";
    write(oss.str());
}