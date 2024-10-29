#pragma once

#if defined(__GNUC__)
    #include <pthread.h>
    #define THREAD_HANDLE pthread_t
    #define THREAD_FUNCTION void*
#else
	#include <windows.h>
    #define THREAD_HANDLE HANDLE
    #define THREAD_FUNCTION  DWORD WINAPI
#endif


class COffThreadMachine
{
public:
	COffThreadMachine(const char* tName); // thread name
	virtual ~COffThreadMachine();

    void SetCPU(int cpu) { cpuToUse = cpu; }
	int StartThread(const char* tName = NULL);
	void StopThread();
	bool IsMachineOn() const { return threadOn; }

protected:
	char threadName[16];

private:
	static THREAD_FUNCTION MachineThread(void*);	

private: 
	// overide these 2 members:
	virtual void ThreadRunning() = 0; // the body to run
	virtual void DoStopThread(); // Called by Stop(), makes extra signal to stop ThreadRunning() if not using IsMachineOn().

private:
	bool threadOn;
	THREAD_HANDLE threadHandle;
    int cpuToUse;  // cpu number is 0-based in CPU_SET. -1: not setting affinity
	
};
