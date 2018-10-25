

#include <cuda_runtime_api.h>

//void setDerivativeParameters();
//extern "C" void setDerivativeParameters(dim3 size, float dx, float dy, float dz);
//extern void FD_Init3dSpaceCos(float* space, int dim, int dim_x, int dim_y, int dim_z, float amplitude, float freq);
//void checkResults(double &error, double &maxError, float *sol, float *df);

void runTest(int dimension);

extern int SimulationSpace_Create(dim3* sim_size);
extern int SimulationSpace_Destroy(void);
extern void SimulationSpace_Timestep(void);
extern int FD_zlineInsert(char* zline, int x, int y, int z, int len);

extern int FD_Testbed(void* image, int sx, int sy, int sz);
