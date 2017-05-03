#ifndef _LAYER
#define _LAYER 1
#include "frect.h"

int LAYER_ReadPng(char* file_name);
void LAYER_ProcessOutline(fRect* dest, indexSize_t index );
void LAYER_Done();




#endif
