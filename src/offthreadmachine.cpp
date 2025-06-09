#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "offthreadmachine.h"

COffThreadMachine::COffThreadMachine(const char* tName)
:threadOn(0), threadHandle(0), cpuToUse(-1)
{
	if(tName) { 
		strncpy(threadName, tName, 63);
		threadName[63] = 0; // Ensure null termination
	}
	else threadName[0] = 0;
}

COffThreadMachine::~COffThreadMachine()
{
	StopThread();
}

/*
   Return CAPNAV_ERROR_SUCCESS if thread started succeessfully; CAPNAV_ERROR_ALEADY_EXISTS if thread is already started,
   CAPNAV_ERROR_CREATE if fail to creat thread object.
*/
int COffThreadMachine::StartThread(const char* tName)
{
	if(threadHandle) return EEXIST;
	if(tName) strncpy(threadName, tName, 16);

#if defined(__GNUC__)
	if(pthread_create(&threadHandle, NULL, MachineThread, (void*)this) != 0)
#else		
	if(! (threadHandle = CreateThread(NULL, 0, MachineThread, (void*)this, 0, NULL)))
#endif
	{
		return ECHILD;
	}
#if defined(__GNUC__)
	pthread_setname_np(threadHandle, threadName);
#else
    //SetThreadDescription(threadHandle, threadName);  // SetThreadDescription is not found.
#endif
	return 0;
}

void COffThreadMachine::StopThread()
{
	if(threadHandle && threadOn)
	{
		threadOn = 0;
		DoStopThread();
#if defined(__GNUC__)
		pthread_join(threadHandle, NULL); 
#else			
		WaitForSingleObject(threadHandle,INFINITE);
		CloseHandle(threadHandle);
#endif			
	}
    threadOn = 0;
    threadHandle = 0;
}

void COffThreadMachine::DoStopThread()
{

}

THREAD_FUNCTION COffThreadMachine::MachineThread(void* arg)
{
	COffThreadMachine* self = (COffThreadMachine*)arg;

	if(self->cpuToUse >= 0) //cpuToUse is 0-based.
    {
#if defined(__GNUC__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(self->cpuToUse, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
        SetThreadAffinityMask(GetCurrentThread(), ((unsigned long long) 1) << self->cpuToUse);
#endif

#ifdef CAPTURE
        LOG("Thread: %s, Setting CPU to %d.\n", self->threadName, self->cpuToUse);
#else
		
#endif

    }
	self->threadOn = 1;
	self->ThreadRunning();
	self->threadOn = 0;
	return NULL;
}

