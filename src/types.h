#ifndef ORANGE_TYPES
#define ORANGE_TYPES
#include <stdint.h>

typedef double f64;
typedef float f32;
typedef uint32_t u32;
typedef uint16_t u16;

struct triple_f{
    f32 x; 
    f32 y;
    f32 z;
};

struct triple_d{
    f64 x; 
    f64 y;
    f64 z;
};

struct tuple_f{
    f32 x;
    f32 y;
};

struct tuple_d{
    f64 x;
    f64 y;
};

struct tuple_i{
    int x;
    int y;
};

#endif
