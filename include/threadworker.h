#pragma once
#include <queue>
#include "offthreadmachine.h"
#include "genericmutex.h"

using namespace std; 

class CThreadWorker : public COffThreadMachine
{
public:
    CThreadWorker(const char* name); // name is the thread name
    virtual ~CThreadWorker();

    void SetID(int i) { id = i; }
    int GetID() { return id; }

    void Reset();
    int GetMyWork() { return myWork; }  // for statistics
   
    void PutObjectToQueueIn(void* f);
    void GetObjectsFromQueueOut(void** f, int* count);

    void* GetObjectFromQueueIn();
    void PutObjectToQueueOut(void* f);

    int GetCountQueueInSize();
    int GetCountQueueOutSize();
    
    // easy way to get counts without using mutext
    int GetCountQueueIn() { return countQueueIn; }  
    int GetCountQueueOut() { return countQueueOut; }
    int GetCountInTotal() { return countInTotal; }  // total count in
    int GetCountOutTotal() {return countOutTotal; } // total count out
    int GetCountQueueInMax() { return countQueueInMax; }

    void SetInterval(unsigned int i) { 
        interval = i; 
#ifdef _WIN32
        intervalMilliSeconds = interval / 1000;
        if(interval % 1000) intervalMilliSeconds++;
#endif
    }


private: 
	virtual void ThreadRunning(); // overides of COffThreadMachine for worker thread

private:
    void ResetInner();
   
    virtual bool WorkerFunction(void* f) { return false; }; // the worker function sub-class
    virtual void WorkerReset() {}; // worker's reset, called by Reset, 

private:
    int id;
    CGenericMutex mutexQueueIn; // protect mutexQueueIn between threads
    std::queue<void*> queueIn;	
    CGenericMutex mutexQueueOut; // protect queueOut between threads
    std::queue<void*> queueOut;	

    // tracing counts
    int countQueueIn; // size of queue in;
    int countQueueOut; // size of queue in;
    int countInTotal;  // count frames put into saver by grab thread calling PutFrameToQueueToSave()
    int countOutTotal;  // count frames get out of saver  by grab thread calling GetFramesFromQueueToDriver()
    int countQueueInMax; // maximum of queue in

//    int cpu;
    int myWork;
    unsigned int interval; // interval of checking queue in micro seconds.
#ifdef _WIN32
    unsigned int intervalMilliSeconds; // interval in Windows, minimum 1 millisecond sleep in Windows, converted from interval, minimum 1 millisecond.
#endif
};