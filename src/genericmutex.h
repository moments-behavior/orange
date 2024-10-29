#pragma once

#ifdef _WIN32
#include <Windows.h>

class CGenericMutex
{
public:
    CGenericMutex(void) { InitializeCriticalSection(&critSec); }
    ~CGenericMutex(void) {DeleteCriticalSection(&critSec);}

    inline void Lock(unsigned int timeOut = INFINITE) { EnterCriticalSection(&critSec);  }

    inline void Unlock() { LeaveCriticalSection(&critSec);}

private:
    CRITICAL_SECTION critSec;
};

#else
#include <pthread.h>
class CGenericMutex
{
public:
    CGenericMutex(void) { pthread_mutex_init(&mutex, NULL); }
    ~CGenericMutex(void) { pthread_mutex_destroy(&mutex); }

    inline void Lock(unsigned int timeOut = -1) { pthread_mutex_lock(&mutex); }

    inline void Unlock() { pthread_mutex_unlock(&mutex); }

private:
    pthread_mutex_t mutex;
};
#endif