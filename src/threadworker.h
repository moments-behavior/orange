#pragma once

#include <queue>
#include <vector>
#include "offthreadmachine.h"
#include "genericmutex.h"
#include <cstdio>

#if defined(__GNUC__)
#include <unistd.h>
#endif

template<typename T>
class CThreadWorker : public COffThreadMachine
{
public:
    CThreadWorker(const char* name);
    virtual ~CThreadWorker();

    void SetID(int i) { id = i; }
    int GetID() const { return id; }

    void Reset();
    int GetMyWork() const { return myWork; }

    // Type-safe methods using templates
    void PutObjectToQueueIn(T* f);
    void GetObjectsFromQueueOut(std::vector<T*>& items);
    T* GetObjectFromQueueOut();
    void PutObjectToQueueOut(T* f);
    T* GetObjectFromQueueIn();

    int GetCountQueueInSize();
    int GetCountQueueOutSize();
    int GetCountQueueIn() const { return countQueueIn; }
    int GetCountQueueOut() const { return countQueueOut; }
    int GetCountInTotal() const { return countInTotal; }
    int GetCountOutTotal() const { return countOutTotal; }
    int GetCountQueueInMax() const { return countQueueInMax; }

    void SetInterval(unsigned int i) {
        interval = i;
#ifdef _WIN32
        intervalMilliSeconds = interval / 1000;
        if(interval % 1000) intervalMilliSeconds++;
#endif
    }

protected:
    // The main worker function that derived classes must implement.
    // It now returns a bool to indicate if the item should be passed to the output queue.
    virtual bool WorkerFunction(T* f) = 0;

    // This function is called when the worker is reset.
    virtual void WorkerReset() {}

private:
    // This overrides the pure virtual function in the base class COffThreadMachine.
    void ThreadRunning() override;

    void ResetInner();

private:
    int id = 0;
    CGenericMutex mutexQueueIn;
    std::queue<T*> queueIn;
    CGenericMutex mutexQueueOut;
    std::queue<T*> queueOut;

    // Tracing counts
    int countQueueIn = 0;
    int countQueueOut = 0;
    int countInTotal = 0;
    int countOutTotal = 0;
    int countQueueInMax = 0;

    int myWork = 0;
    unsigned int interval;
#ifdef _WIN32
    unsigned int intervalMilliSeconds;
#endif
};

// --- Template Implementation ---

template<typename T>
CThreadWorker<T>::CThreadWorker(const char* name)
    : COffThreadMachine(name),
      id(0),
      countQueueIn(0),
      countQueueOut(0),
      countInTotal(0),
      countOutTotal(0),
      countQueueInMax(0),
      myWork(0),
      interval(10)
{
    this->ResetInner();
#ifdef _WIN32
    intervalMilliSeconds = 1;
#endif
}

template<typename T>
CThreadWorker<T>::~CThreadWorker()
{
}

template<typename T>
void CThreadWorker<T>::Reset()
{
    this->ResetInner();
    this->WorkerReset();
}

template<typename T>
void CThreadWorker<T>::ResetInner()
{
    myWork = 0;
    countQueueIn = 0;
    countQueueOut = 0;
    countInTotal = 0;
    countOutTotal = 0;
    countQueueInMax = 0;

    // Safely clear the queues
    mutexQueueIn.Lock();
    std::queue<T*> emptyIn;
    std::swap(queueIn, emptyIn);
    mutexQueueIn.Unlock();

    mutexQueueOut.Lock();
    std::queue<T*> emptyOut;
    std::swap(queueOut, emptyOut);
    mutexQueueOut.Unlock();
}

template<typename T>
void CThreadWorker<T>::PutObjectToQueueIn(T* f)
{
    mutexQueueIn.Lock();
    queueIn.push(f);
    countQueueIn++;
    countInTotal++;
    if (countQueueInMax < countQueueIn) {
        countQueueInMax = countQueueIn;
    }
    mutexQueueIn.Unlock();
}

template<typename T>
void CThreadWorker<T>::GetObjectsFromQueueOut(std::vector<T*>& items)
{
    mutexQueueOut.Lock();
    items.clear();
    while (!queueOut.empty())
    {
        items.push_back(queueOut.front());
        queueOut.pop();
    }
    countOutTotal += static_cast<int>(items.size());
    countQueueOut = 0; // The queue is now empty
    mutexQueueOut.Unlock();
}

template<typename T>
T* CThreadWorker<T>::GetObjectFromQueueOut()
{
    T* f = nullptr;
    mutexQueueOut.Lock();
    if (!queueOut.empty())
    {
        f = queueOut.front();
        queueOut.pop();
        countQueueOut--;
    }
    mutexQueueOut.Unlock();
    return f;
}


template<typename T>
void CThreadWorker<T>::PutObjectToQueueOut(T* f)
{
    mutexQueueOut.Lock();
    queueOut.push(f);
    countQueueOut++;
    mutexQueueOut.Unlock();
}

template<typename T>
int CThreadWorker<T>::GetCountQueueInSize()
{
    int size = -1;
    mutexQueueIn.Lock();
    size = static_cast<int>(queueIn.size());
    mutexQueueIn.Unlock();
    return size;
}

template<typename T>
int CThreadWorker<T>::GetCountQueueOutSize()
{
    int size = -1;
    mutexQueueOut.Lock();
    size = static_cast<int>(queueOut.size());
    mutexQueueOut.Unlock();
    return size;
}

template<typename T>
T* CThreadWorker<T>::GetObjectFromQueueIn()
{
    T* f = nullptr;
    mutexQueueIn.Lock();
    if (!queueIn.empty())
    {
        f = queueIn.front();
        queueIn.pop();
        countQueueIn--;
    }
    mutexQueueIn.Unlock();
    return f;
}

template<typename T>
void CThreadWorker<T>::ThreadRunning()
{
    printf("Child Thread Start %d\n", id);
    // FIX: Use 'this->' to explicitly qualify member function calls inside a template
    while (this->IsMachineOn())
    {
        T* f = this->GetObjectFromQueueIn();
        if (f)
        {
            if (this->WorkerFunction(f)) // Check the return value
            {
                this->PutObjectToQueueOut(f);
            }
            myWork++;
        }
        else
        {
#if defined(__GNUC__)
            usleep(interval);
#else
            Sleep(intervalMilliSeconds);
#endif
        }
    }

    // Process remaining items in the queue after thread is stopped
    while (true)
    {
        T* f = this->GetObjectFromQueueIn();
        if (f)
        {
            if (this->WorkerFunction(f)) // Check the return value
            {
                 this->PutObjectToQueueOut(f);
            }
            myWork++;
        }
        else
        {
            break;
        }
    }
    printf("Child Thread DONE %d\n", id);
}