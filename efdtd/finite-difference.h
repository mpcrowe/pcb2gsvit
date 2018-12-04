#ifndef _FINITE_DIFFERENCE
#define _FINITE_DIFFERENCE 1

#include <cuda_runtime_api.h>

//void setDerivativeParameters();
//extern "C" void setDerivativeParameters(dim3 size, float dx, float dy, float dz);
//extern void FD_Init3dSpaceCos(float* space, int dim, int dim_x, int dim_y, int dim_z, float amplitude, float freq);
//void checkResults(double &error, double &maxError, float *sol, float *df);

#define MAX_SIZE_MATERIAL_TABLE 128

void runTest(int dimension);

extern int SimulationSpace_Create(dim3* sim_size);
extern int SimulationSpace_Destroy(void);
extern void SimulationSpace_Timestep(void);

extern int FD_zlineInsert(char* zline, int x, int y, int z, int len);

extern int FD_Testbed(void* image, int sx, int sy, int sz);

extern int FD_UpdateMatIndex(char* src, int len, int offset);
extern int FD_UpdateGa(float* ptr, int len);
extern int FD_UpdateGb(float* ptr, int len);
extern int FD_UpdateGc(float* ptr, int len);
extern int FD_UpdateDelExp(float del_exp);
extern void FD_UpdateDeltas(float dx, float dy, float dz);

extern int SimulationSpace_ExtrudeY(char* src, int xDim, int zDim, int xCenter, int zCenter, int yStart, int yLen);
extern int SimulationSpace_ExtrudeZ(char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int Zend);
extern int SimulationSpace_ExtrudeX(char* src, int yDim, int zDim, int yCenter, int zCenter, int xStart, int xLen);
extern int MatIndex_ExtrudeZ(char* dest, int dx, int dy, int dz, char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen);

#endif
