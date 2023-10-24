#include "SyncDisplay.h"

void detection_proc(SyncDisplay* sync_manager, int idx)
{

    while(true) {        
        // wait for frame ready
        printf("wait for kick\n");
        sync_manager->WaitForKick();
        
        printf("detection\n");
        sync_manager->SignalMoveSent(idx);
        // detection, busy 
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        
        printf("detection done \n");
		sync_manager->SignalDetectionDone(idx);
    }

    return;  
}