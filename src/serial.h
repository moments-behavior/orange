#pragma once

#include <string>
#include <vector>
#include <map>

struct PumpConfig {
    float target_uL;
    int dispense_time_ms;
    int cycles;
    int delay;
    float syringe_ID_mm;
    int steps_per_rev;
    int microsteps;
    float lead_mm;
    int push_direction;
    int control_mode;
    bool repeat;
    int repeat_delay;
};

static std::map<char, PumpConfig> cfg;

std::vector<std::string> list_json_files_in_folder();
bool load_pump_config(const std::string& filename, std::map<char, PumpConfig>& config);
const std::map<char, PumpConfig>& get_loaded_pump_configs(std::string filename);
void initialize_pump_state_from_config(const std::map<char, PumpConfig>& configs,
    const char pump_ids[],
    float microliters[3],
    int delivery_ms[3],
    int cycles[3],
    int delays[3],
    int push_direction[3],
    int control_mode[3],
    bool repeat[3],
    int repeat_delay[3]);

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

        void send_pump_command(char pump, bool push, float ul, int dispense_time_ms);

    private:
        int fd_;
};