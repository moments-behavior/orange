
#ifndef ORANGE_THREADS
#define ORANGE_THREADS
#include <stdint.h>
#include <time.h>

#ifdef _WIN32 // WINDOWS
#include <windows.h>

// Increment value with a lock and return the previous value
inline uint64_t sync_fetch_and_add(volatile uint64_t* x, uint64_t by) {
	return InterlockedExchangeAdd64((volatile int64_t*)x, by);
}

inline float tick() {
	LARGE_INTEGER current_ticks;
	LARGE_INTEGER tick_frequency;
	BOOL res = QueryPerformanceCounter(&current_ticks);
	if (!res) {
		return 0;
	}
	res = QueryPerformanceFrequency(&tick_frequency);
	return (float)current_ticks.QuadPart / (float)tick_frequency.QuadPart;
}

#else
inline float tick() {
	struct timespec ts;
	uint32_t res = clock_gettime(CLOCK_MONOTONIC, &ts);
	if (res == -1) {
		return 0;
	}
	return ((float) ((ts.tv_sec * 1e9) + ts.tv_nsec)) / (float) 1.0e9;
}


// Increment value with a lock and return the previous value
inline uint64_t sync_fetch_and_add(volatile uint64_t *x, uint64_t by) {
	// NOTE(dd): we're using a gcc/clang compiler extension to do this
	// because mutexes were for some reason slower
	return __sync_fetch_and_add(x, by);
}
#endif

#endif //ORANGE_THREADS



