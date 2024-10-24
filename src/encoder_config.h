#ifndef ORANGE_ENCODER_CONFIG_H
#define ORANGE_ENCODER_CONFIG_H

#include <string>

struct EncoderConfig {
    std::string encoder_basic_setup;  // Will be built dynamically
    std::string encoder_codec;
    std::string encoder_preset;
    std::string folder_name;
    std::string encoder_setup;        // Add this for backwards compatibility

    // Declare the method
    void UpdateEncoderSetup();
};

// Define the method outside the struct
inline void EncoderConfig::UpdateEncoderSetup() {
    encoder_basic_setup = "-codec " + encoder_codec + " -preset " + encoder_preset + " -fps ";
    encoder_setup = encoder_basic_setup;  // Keep encoder_setup updated for compatibility
}

#endif // ORANGE_ENCODER_CONFIG_H