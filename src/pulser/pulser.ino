// Pulse Generator for 32-bit Teensy 

// HHIm Janelia jET sws

#define VERSION 20240429

// 20240429 sws
// - initial program 


#include <Cmd.h>

IntervalTimer pulseWidthTimer;
IntervalTimer pulseOnTimer;

#define MINFREQ 1
#define MAXFREQ 200
#define MINON 20
#define MAXON 1000

int16_t freq = 100;
int16_t onTime = 20;

//
//// =======================
//// ===  P U L S E   O N   I N T  ===
//// =======================

void pulseOnInt()
{
   pulseOnTimer.end();
   digitalWriteFast(0, LOW);
   digitalWriteFast(1, LOW);
   digitalWriteFast(2, LOW);
   digitalWriteFast(3, LOW);
}


// =======================
// ===  P U L S E   W I D T H   I N T  ===
// =======================

void pulseWidthInt()
{
   pulseOnTimer.begin(pulseOnInt, onTime);
   digitalWriteFast(0, HIGH);
   digitalWriteFast(1, HIGH);
   digitalWriteFast(2, HIGH);
   digitalWriteFast(3, HIGH);
}


// =======================
// ===  F R E Q   C M D  ===
// =======================

void freqCmd(int arg_cnt, char **args)
{

  if ( arg_cnt > 1 )
  {
    int16_t newFreq = cmdStr2Num(args[1], 10);
    if( (newFreq >= MINFREQ) && (newFreq <= MAXFREQ) )
    {
      freq = newFreq;
      pulseWidthTimer.update(1e6/freq);    
  
    }
    
  }
}

// =======================
// ===  O N   T I M E   C M D  ===
// =======================

void onTimeCmd(int arg_cnt, char **args)
{
  if ( arg_cnt > 1 )
  {
    
    int16_t newOnTime = cmdStr2Num(args[1], 10);
    if( (newOnTime >= MINON) && (newOnTime <= MAXON) )
    {
       if( newOnTime < 1e6/freq ) 
          onTime = newOnTime;      
    }   
  }
}

// =======================
// ===  S T A R T   C M D  ===
// =======================

void startCmd(int arg_cnt, char **args)
{
    pulseWidthTimer.begin( pulseWidthInt, 1e6/freq);
}


// =======================
// ===  S T O P   C M D  ===
// =======================

void stopCmd(int arg_cnt, char **args)
{
    pulseWidthTimer.end();
}

// =======================
// ===  L I S T  C M D  ===
// =======================

void listCmd(int arg_cnt, char **args)
{
    Serial.println(VERSION);
    cmdList();
}

void setup() 
{
   pinMode(0, OUTPUT);
   pinMode(1, OUTPUT);
   pinMode(2, OUTPUT);
   pinMode(3, OUTPUT);
   digitalWriteFast(0, LOW);
   digitalWriteFast(1, LOW);
   digitalWriteFast(2, LOW); 
   digitalWriteFast(3, LOW); 
   Serial.begin(9600);
   cmdInit(&Serial);
   cmdAdd("FREQ", freqCmd);
   cmdAdd("ON", onTimeCmd);    
   cmdAdd("START", startCmd);
   cmdAdd("STOP", stopCmd);
   cmdAdd("???", listCmd);
//   pulseWidthTimer.begin( pulseWidthInt, 1e6/freq);
}

void loop() 
{
   cmdPoll();
}