
#ifndef ORANGE_THREADS
#define ORANGE_THREADS
#include <stdint.h>
#include <time.h>

inline float tick() {
	struct timespec ts;
	uint32_t res = clock_gettime(CLOCK_MONOTONIC, &ts);
	if (res == -1) {
		return 0;
	}
	return ((float) ((ts.tv_sec * 1e9) + ts.tv_nsec)) / (float) 1.0e9;
}

#endif //ORANGE_THREADS
