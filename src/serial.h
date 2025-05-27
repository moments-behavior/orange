#pragma once

#include <string>
#include <vector>


class SerialPort {
    public:
        SerialPort();
        ~SerialPort();
    
        static std::vector<std::string> list_available_ports();
    
        bool open(const std::string& port_name, int baud_rate = 9600);
        void close();
        bool is_open() const;
    
        bool write(const std::string& data);
        std::string read();
    
    private:
        int fd_; // File descriptor for Linux
};