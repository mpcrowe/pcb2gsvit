

#include <cuda_runtime_api.h>

void setDerivativeParameters();
//void checkResults(double &error, double &maxError, float *sol, float *df);
void runTest(int dimension);
extern int SimulationSpace_Create(dim3* sim_size);
extern int SimulationSpace_Destroy(void);
