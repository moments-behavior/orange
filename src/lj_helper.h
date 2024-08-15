#pragma once
#ifndef LOGHELPER
#define LOGHELPER

#include <thread>

// pulse data structure
struct Pulse    {
    const double highVoltage = 5.0; // High level of the PWM signal
    const double lowVoltage = 0.0;  // Low level of the PWM signal
    double frequency = 60.0;        // Target frequency in Hz
    double dutyCycle = 15.0;        // Duty cycle in percent

    // Calculate high and low durations based on frequency and duty cycle
    double period = 1.0 / frequency; // Period of the PWM signal in seconds
    double highTime = period * (dutyCycle / 100.0); // High state duration
    double lowTime = period - highTime; // Low state duration

    // Convert durations to microseconds for usleep
    int highTime_us = (int)(highTime * 1e6);
    int lowTime_us = (int)(lowTime * 1e6);

    uint64_t counter = 0;
};

struct LabJackState{
    // states
    bool is_connected=false;
    bool pulse_on=false;
    int handle;
    int err;   

};

extern void update_pulse(Pulse* pulse, double frequency, double dutyCycle);

extern void pulse_on(LabJackState* lj_state, Pulse* pulse);

extern void open_labjack(LabJackState* lj_state);

extern void close_labjack(LabJackState* lj_state);

extern void start_pulsing(LabJackState* lj_state, Pulse* pulse);

extern void stop_pulsing(LabJackState* lj_state); 

#endif