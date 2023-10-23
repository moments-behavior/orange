#include "SynchronizedDisplay.h"

void detection_proc(SyncDisplay* sync_manager, int idx)
{

    while(true) {        
        // wait for frame ready
        sync_manager->WaitForKick();

        // detection, busy 
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        
        // 
		sync_manager->SignalDetectionDone(idx);
    }

    return;  
}