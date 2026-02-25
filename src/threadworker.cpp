#include <stdio.h>
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "threadworker.h"

CThreadWorker::CThreadWorker(const char *name)
    : COffThreadMachine(name), interval(10) {
    ResetInner();
#ifdef _WIN32
    intervalMilliSeconds = 1;
#endif
}

CThreadWorker::~CThreadWorker() {}

/*
  Reset CThreadWorker by reseting frame buffer.
  This is to avoid the frame from requeued to NIC drvier when swapped out by
  SaveFrame() at the beginning of save frame.
*/
void CThreadWorker::Reset() {
    ResetInner();
    WorkerReset();
};

void CThreadWorker::ResetInner() {
    myWork = 0;
    countQueueIn = countQueueOut = 0;
    countInTotal = countOutTotal = 0;
    countQueueInMax = 0;
}

void *CThreadWorker::GetObjectFromQueueIn() {
    void *f = NULL;
    mutexQueueIn.Lock();
    if (queueIn.size() > 0) // check if there is available thumbnail
    {
        f = queueIn.front();
        queueIn.pop();
        countQueueIn--;
    }
    mutexQueueIn.Unlock();
    return f;
}

// interface to extenal
void CThreadWorker::PutObjectToQueueIn(void *f) {
    mutexQueueIn.Lock();
    queueIn.push(f);
    countQueueIn++;
    countInTotal++;
    mutexQueueIn.Unlock();
    // countQueueInMax is statistic, no need to use mutex
    if (countQueueInMax < countQueueIn)
        countQueueInMax = countQueueIn;
}

// interface to extenal
/*
  Get multiple frames from queue to driver.
  We don't know how many items in this queue. It could be more than
  FRAMES_PER_SAVER although we check this number when calling
  PutFrameToQueueToSave(). So count points to the size of f when passed in.
  Value changed to the actual number when function returns. f is a pointer to
  pointes arrary
*/
void CThreadWorker::GetObjectsFromQueueOut(void **f, int *count) {
    mutexQueueOut.Lock(); // check if there is available thumbnail
    int size = queueOut.size();
    if (size < *count) {
        *count = size;
    }
    for (int i = 0; i < *count; i++) {
        f[i] = queueOut.front();
        queueOut.pop();
    }
    countOutTotal += *count;
    countQueueOut -= *count;
    ;
    mutexQueueOut.Unlock();
}

void CThreadWorker::PutObjectToQueueOut(void *f) {
    mutexQueueOut.Lock();
    queueOut.push(f);
    countQueueOut++;
    mutexQueueOut.Unlock();
}

int CThreadWorker::GetCountQueueInSize() {
    int size = -1;
    mutexQueueIn.Lock();
    size = queueIn.size();
    mutexQueueIn.Unlock();
    return size;
}

int CThreadWorker::GetCountQueueOutSize() {
    int size = -1;
    mutexQueueOut.Lock();
    size = queueOut.size();
    mutexQueueOut.Unlock();
    return size;
}

void CThreadWorker::ThreadRunning() {
    printf("Child Thread Start %d\n", id);
    while (IsMachineOn()) {
        void *f = GetObjectFromQueueIn();
        if (f) {
            WorkerFunction(f);
            PutObjectToQueueOut(f);
            myWork++;
        } else {
#if defined(__GNUC__)
            usleep(interval);
#else
            Sleep(intervalMilliSeconds);
#endif
        }
    }

    printf("Child Thread DONE %d\n", id);
}
