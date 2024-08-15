


#include "LJM_Utilities.h"
#include "LJM_StreamUtilities.h"

#include "lj_helper.h"


const char * identifier = "ANY";
bool clean_exit = false;

std::thread* pulse_on_thread;


void pulse_on(LabJackState* lj_state, Pulse* pulse)
{
    printf("pulsing now\n");
    while(lj_state->pulse_on)
    {
        // Set DAC0 to high voltage
        LJM_eWriteName(lj_state->handle, "DAC0", pulse->highVoltage);
        std::this_thread::sleep_for(std::chrono::microseconds(pulse->highTime_us)); // Wait for high state duration
      
        // Set DAC0 to low voltage
        LJM_eWriteName(lj_state->handle, "DAC0", pulse->lowVoltage);
        std::this_thread::sleep_for(std::chrono::microseconds(pulse->lowTime_us)); // Wait for high state duration

    }

    clean_exit = true;
}

void update_pulse(Pulse* pulse, double frequency, double dutyCycle) {
    pulse->frequency = frequency;
    pulse->dutyCycle = dutyCycle;
    pulse->period = 1.0 / pulse->frequency;
    pulse->highTime = pulse->period * (pulse->dutyCycle / 100.0);
    pulse->lowTime = pulse->period - pulse->highTime;
    pulse->highTime_us = (int)(pulse->highTime * 1e6);
    pulse->lowTime_us = (int)(pulse->lowTime * 1e6);
}

void open_labjack(LabJackState* lj_state)       {
    lj_state->err = LJM_Open(LJM_dtT7, LJM_ctETHERNET, identifier, &lj_state->handle);
    if(lj_state->err!=LJME_NOERROR)   {
        lj_state->is_connected=false;
        printf("Failed to connect to LabJack\n");
        return;
    }
    else        {
        lj_state->is_connected=true;
        printf("Opened connection to LabJack\n");            
    }

    
}

void close_labjack(LabJackState* lj_state)      {
    lj_state->err = LJM_Close(lj_state->handle);
    if(lj_state->err!=LJME_NOERROR)   {
        printf("Failed to close LabJack\n");
        return;
    }
    else        {
        lj_state->is_connected=false;
        printf("Closed connecttion to LabJack\n");
        GetCurrentTimeMS();
        LJM_GetHostTick();
    }
    
}

void start_pulsing(LabJackState* lj_state, Pulse* pulse)    {

    lj_state->pulse_on = true;
    pulse_on_thread = new std::thread(pulse_on,lj_state,pulse);
    printf("Started pulsing\n");
    clean_exit = false;
    
}

void stop_pulsing(LabJackState* lj_state)     {
    
    lj_state->pulse_on = false;

    while(clean_exit==false)
    {
        printf("Waiting for clean exit on pulse thread\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    clean_exit = false;
    pulse_on_thread->join();    
    delete pulse_on_thread;
    pulse_on_thread = nullptr;
    printf("Stopped pulsing\n");
    
}

