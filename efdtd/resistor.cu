#include <stdio.h>
#include <assert.h>
extern "C" {
#include "finite-difference.h"
}

#define DEBUG 1
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result, int lineNum)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: (%s) %d at line %d\n", cudaGetErrorString(result), result, lineNum);
		//      assert(result == cudaSuccess);
		exit(-1);
	}
#endif
	return(result);
}


struct resistor
{
	char* c_mi;
	int mx;
	int my;
	int mz;
	int bytes;
};

struct resistor* RES_new(int dimX, int dimY, int dimZ)
{
	//create new structure;
	struct resistor* res = (struct resistor*)malloc(sizeof(struct resistor));
	if(res == NULL)
		return(NULL);
	res->mx = dimX;
	res->my = dimY;
	res->mz = dimZ;
	res->bytes = res->mx*res->my*res->mz * sizeof(char);

	//   allocate space in GPU;
        checkCuda( cudaMalloc((void**)&res->c_mi, res->bytes), __LINE__  );
        return(res);
}


void RES_Destroy(struct resistor* res)
{
	if(res == NULL)
		return;
	checkCuda( cudaFree(res->c_mi), __LINE__  );
	free(res);		
}


void RES_MakeChip(struct resistor* res, int padX, int padY, int height, float val)
{
	int xOff = res->mx/2;
	int yOff = res->my/2;
	//	extrude left pad from copper
	char* pad = (char*)malloc(padX*padY*sizeof(char));
	memset(pad,3,padX*padY);
	MatIndex_ExtrudeZ(res->c_mi, res->mx, res->my, res->mz, pad, padX, padY, xOff-padX/2, yOff-padY/2, 0, height);
	//	extrude right pad from copper
	MatIndex_ExtrudeZ(res->c_mi, res->mx, res->my, res->mz, pad, padX, padY, xOff+padX/2, yOff+padY/2, 0, height);

	//	extrude ceramic botomm non-conductor
	free(pad);
	int baseX = res->mx-2*padX;
	int baseY = padY;
	pad = (char*)malloc(baseX*baseY*sizeof(char));
	memset(pad,4,baseX*baseY);
	MatIndex_ExtrudeZ(res->c_mi, res->mx, res->my, res->mz, pad, baseX, baseY, xOff, yOff, 0, height-1);

	//	compute conductivity value for resistor "surface"
	memset(pad,4,baseX*baseY);
	//	extrude resistive surface
	MatIndex_ExtrudeZ(res->c_mi, res->mx, res->my, res->mz, pad, baseX, baseY, xOff, yOff, height-1, height);
}


