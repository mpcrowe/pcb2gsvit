#ifndef _FILE_PROCESSING
#define _FILE_PROCESSING 1

#include <errno.h>
extern  int FP_ReadRiff(char* riffFname);
extern int FP_ProcessFile(char* fname, int verbose, int silent);
extern void FP_MakeVia(int xCenter, int yCenter, int outerRadius, int innerRadius, int zStart, int zLen, char matIndex);
extern void FP_MakeRectangleZ(int xCenter, int yCenter, int xLen, int yLen, int zStart, int zLen, char matIndex);

#endif
