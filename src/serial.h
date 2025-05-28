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
    
        void send_pump_command(char pump, bool push, int cycles, int delay_us);

        void send_pump_command(char pump, bool push, float ul);

    private:
        int fd_;
};