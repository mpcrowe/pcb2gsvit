#ifndef _MED_VECT
#define MED_VECT 1
// med_vect
// processes an xml drill fragment to produce a medium vector file format
// of cylinders


#include <stdio.h>
#include <string.h>
#include "xpu.h"

// allocates memory for a new frect instance
FILE* MV_Open(char* name);
void MV_Close(FILE* mvfd);
int MV_ProcessDrillNodeSet(FILE* mvfd, xmlNodeSetPtr xnsPtr, int zstart, int zstop);


#endif
