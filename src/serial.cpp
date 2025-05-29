#include "serial.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <vector>
#include <dirent.h>
#include <cstring>
#include "json.hpp"
#define PI 3.14159265358979323846
using json = nlohmann::json;


SerialPort::SerialPort() : fd_(-1) {}
SerialPort::~SerialPort() { close(); }

std::vector<std::string> list_json_files_in_folder() {
    const std::string folder_path = "/home/user/orange_data/config/pump";
    std::vector<std::string> result;

    DIR* dir = opendir(folder_path.c_str());
    if (!dir) return result;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);
        if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json") {
            result.push_back(folder_path + "/" + filename); 
        }
    }

    closedir(dir);
    return result;
}


bool load_pump_config(const std::string& filename, std::map<char, PumpConfig>& config) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "failed to open config\n";
        return false;
    }

    try {

        json j;
        in >> j;

        cfg.clear(); // make room for new config

        for (const auto& [key, val] : j.items()) {
            char pump_id = key[0];
            PumpConfig cfg;
            cfg.target_uL = val.at("target_uL").get<float>();
            cfg.dispense_time_ms = val.at("dispense_time_ms").get<int>();
            cfg.cycles = val.at("cycles").get<int>();
            cfg.delay = val.at("delay").get<int>();
            cfg.syringe_ID_mm = val.at("syringe_ID_mm").get<float>();
            cfg.steps_per_rev = val.at("steps_per_rev").get<int>();
            cfg.microsteps = val.at("microsteps").get<int>();
            cfg.lead_mm = val.at("lead_mm").get<float>();
            cfg.push_direction = val.at("push_direction").get<int>();
            cfg.control_mode = val.at("control_mode").get<int>();
            cfg.repeat = val.at("repeat").get<bool>();
            cfg.repeat_delay = val.at("repeat_delay").get<int>();
            config[pump_id] = cfg;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing config: " << e.what() << "\n";
        return false;
    }

    return true;
}


const std::map<char, PumpConfig>& get_loaded_pump_configs(std::string filename) {
    std::string user = std::getenv("USER");
    if (!load_pump_config(filename, cfg)) {
        std::cerr << "failed to load config\n";
    }
    return cfg;
}

void initialize_pump_state_from_config(const std::map<char, PumpConfig>& configs,
    const char pump_ids[],
    float microliters[3],
    int delivery_ms[3],
    int cycles[3],
    int delays[3],
    int push_directions[3],
    int control_mode[3],
    bool repeat[3],
    int repeat_delay[3]) {
    for (int i = 0; i < 3; ++i) {
        char id = pump_ids[i];
        if (configs.count(id)) {
            const auto& cfg = configs.at(id);
            microliters[i] = cfg.target_uL;
            delivery_ms[i] = cfg.dispense_time_ms;
            cycles[i] = cfg.cycles;
            delays[i] = cfg.delay;
            push_directions[i] = cfg.push_direction;
            control_mode[i] = cfg.control_mode;
            repeat[i] = cfg.repeat;
            repeat_delay[i] = cfg.repeat_delay;
        }
    }
}

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



void SerialPort::send_pump_command(char pump, bool push, float ul, int dispense_time_ms) {
    if (!is_open()) return;

    std::ostringstream oss;

    PumpConfig config = cfg[pump];

    double lead_mm = config.lead_mm;
    int steps_per_rev = config.steps_per_rev;
    int microsteps = config.microsteps;
    double syringe_ID_mm = config.syringe_ID_mm;

    int usteps_per_rev = steps_per_rev * microsteps;
    double usteps_per_mm = usteps_per_rev / lead_mm;

    double r = syringe_ID_mm / 2.0;
    double stroke = ul / (PI * r * r);

    int pulse_count = (int) (stroke * usteps_per_mm);

    int delay = dispense_time_ms * 1000 / pulse_count;
    
    oss << (push ? 'h' : 'l') << pump << " " << pulse_count << " " << delay << "\n";
    write(oss.str());

}