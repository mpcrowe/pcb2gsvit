/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/* 
*based on 
* Mark Harris March 2013 "Finite Difference Methods in CUDA C/C++, Part 1"
*for details see
*https://devblogs.nvidia.com/finite-difference-methods-cuda-cc-part-1
*/



#include <stdio.h>
#include <assert.h>
extern "C" {
#include "finite-difference.h"
}
#define DEBUG 1

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

float fx = 6.0f, fy = 1.0f, fz = 1.0f;
const int mx = 256, my = 256, mz = 256;
__constant__ int c_mx, c_my, c_mz;

// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to
//     calculate the derivative at mutiple points
const int sPencils = 4;  // small # pencils

dim3 numBlocks[3][2], threadsPerBlock[3][2];

// stencil coefficients
__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_axn, c_bxn, c_cxn, c_dxn;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;

__constant__ float c_coef_x[9];


static int setDerivativeParametersX(int voxels, float scale)
{
	float dsinv = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len x' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float ax =  4.f / 5.f   * dsinv;
	float bx = -1.f / 5.f   * dsinv;
	float cx =  4.f / 105.f * dsinv;
	float dx = -1.f / 280.f * dsinv;
	checkCuda( cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	ax = -ax;
	bx = -bx;
	cx =  -cx;
	dx = -dx;
	checkCuda( cudaMemcpyToSymbol(c_axn, &ax, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_bxn, &bx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_cxn, &cx, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_dxn, &dx, sizeof(float), 0, cudaMemcpyHostToDevice) );

	float coef_x[9];
	coef_x[0] = (1.f  / 280.f) * dsinv;	// -dx
	coef_x[1] = (-4.f / 105.f) * dsinv;	// -cx
	coef_x[2] = (1.f  / 5.f)   * dsinv;	// -bx
	coef_x[3] = (-4.f / 5.f)   * dsinv;	// -ax
	coef_x[4] = 0.f;
	coef_x[5] = (4.f  / 5.f)   * dsinv;	// ax
	coef_x[6] = (-1.f / 5.f)   * dsinv;	// bx
	coef_x[7] = (4.f  / 105.f) * dsinv;	// cx
	coef_x[8] = (-1.f / 280.f) * dsinv;	// dx
	checkCuda( cudaMemcpyToSymbol(c_coef_x, coef_x, 9* sizeof(float)) );

	checkCuda( cudaMemcpyToSymbol(c_mx, &voxels, sizeof(int), 0, cudaMemcpyHostToDevice) );
	return (0);
}

static int setDerivativeParametersY(int voxels, float scale)
{
	float dsinv = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len y' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float ay =  4.f / 5.f   * dsinv;
	float by = -1.f / 5.f   * dsinv;
	float cy =  4.f / 105.f * dsinv;
	float dy = -1.f / 280.f * dsinv;
	checkCuda( cudaMemcpyToSymbol(c_ay, &ay, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_by, &by, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_cy, &cy, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_dy, &dy, sizeof(float), 0, cudaMemcpyHostToDevice) );


	checkCuda( cudaMemcpyToSymbol(c_my, &voxels, sizeof(int), 0, cudaMemcpyHostToDevice) );
	return (0);
}

static int setDerivativeParametersZ(int voxels, float scale)
{
	float dsinv = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len Z' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float az =  4.f / 5.f   * dsinv;
	float bz = -1.f / 5.f   * dsinv;
	float cz =  4.f / 105.f * dsinv;
	float dz = -1.f / 280.f * dsinv;
	checkCuda( cudaMemcpyToSymbol(c_az, &az, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_bz, &bz, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_cz, &cz, sizeof(float), 0, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpyToSymbol(c_dz, &dz, sizeof(float), 0, cudaMemcpyHostToDevice) );

	checkCuda( cudaMemcpyToSymbol(c_mz, &voxels, sizeof(int), 0, cudaMemcpyHostToDevice) );
	return (0);
}

// host routine to set constant data
extern "C" void setDerivativeParameters()
{
	setDerivativeParametersX(mx, 1.0);
	setDerivativeParametersY(my, 1.0);
	setDerivativeParametersZ(mz, 1.0);

	// Execution configurations for small pencil tiles

	// x derivative
	numBlocks[0][0]  = dim3(my / sPencils, mz, 1);
	threadsPerBlock[0][0] = dim3(mx, sPencils, 1);
	numBlocks[0][1]  = dim3(my / sPencils, mz, 1);
	threadsPerBlock[0][1] = dim3(mx, sPencils, 1);

	// y derivative
	numBlocks[1][0]  = dim3(mx / sPencils, mz, 1);
	threadsPerBlock[1][0] = dim3(sPencils, my, 1);
	numBlocks[1][1]  = dim3(mx / sPencils, mz, 1);
	threadsPerBlock[1][1] = dim3(sPencils, my, 1);

	// z derivative
	numBlocks[2][0]  = dim3(mx / sPencils, my, 1);
	threadsPerBlock[2][0] = dim3(sPencils, mz, 1);
	numBlocks[2][1]  = dim3(mx / sPencils, my, 1);
	threadsPerBlock[2][1] = dim3(sPencils, mz, 1);
}


void initInput(float *f, int dim)
{
	const float twopi = 8.f * (float)atan(1.0);

	for (int k = 0; k < mz; k++)
	{
		for (int j = 0; j < my; j++)
		{
			for (int i = 0; i < mx; i++)
			{
				switch (dim)
				{
				case 0:
					f[k*mx*my+j*mx+i] = cos(fx*twopi*(i-1.f)/(mx-1.f));
				break;
				case 1:
					f[k*mx*my+j*mx+i] = cos(fy*twopi*(j-1.f)/(my-1.f));
				break;
				case 2:
					f[k*mx*my+j*mx+i] = cos(fz*twopi*(k-1.f)/(mz-1.f));
				break;
				}
			}
		}
	}
}


void initSol(float *sol, int dim)
{
	const float twopi = 8.f * (float)atan(1.0);

	for (int k = 0; k < mz; k++)
	{
		for (int j = 0; j < my; j++)
		{
			for (int i = 0; i < mx; i++)
			{
				switch (dim)
				{
				case 0:
					sol[k*mx*my+j*mx+i] = -fx*twopi*sin(fx*twopi*(i-1.f)/(mx-1.f));
				break;
				case 1:
					sol[k*mx*my+j*mx+i] = -fy*twopi*sin(fy*twopi*(j-1.f)/(my-1.f));
				break;
				case 2:
					sol[k*mx*my+j*mx+i] = -fz*twopi*sin(fz*twopi*(k-1.f)/(mz-1.f));
				break;
				}
			}
		}
	}
}


void checkResults(double &error, double &maxError, float *sol, float *df)
{
	// error = sqrt(sum((sol-df)**2)/(mx*my*mz))
	// maxError = maxval(abs(sol-df))
	maxError = 0;
	error = 0;
	for (int k = 0; k < mz; k++)
	{
		for (int j = 0; j < my; j++)
		{
			for (int i = 0; i < mx; i++)
			{
				float s = sol[k*mx*my+j*mx+i];
				float f = df[k*mx*my+j*mx+i];
				//printf("%d %d %d: %f %f\n", i, j, k, s, f);
				error += (s-f)*(s-f);
				if (fabs(s-f) > maxError) maxError = fabs(s-f);
			}
		}
	}
	error = sqrt(error / (mx*my*mz));
}


// -------------
// x derivatives
// -------------
// this function takes the derivative with respect to the x axis and
// accumulates the result in to the destination array
// this is use full when computing the curl of a field
__global__ void derivativeAccumX(float *f, float *df)
{
	extern __shared__ float s_f[]; // 4-wide halo

	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int k  = blockIdx.y;
	int si = i + 4;       // local i for shared memory access + halo offset
	int sj = (c_mx+8)*threadIdx.y; // local j for shared memory access

	int globalIdx = k * c_mx * my + j * c_mx + i;

	s_f[sj+si] = f[globalIdx];

	__syncthreads();

	// fill in periodic images in shared memory array
	if (i < 4)
	{
		s_f[sj+si-4]  = s_f[sj+si+c_mx-5];
		s_f[sj+si+c_mx] = s_f[sj+si+1];
	}

	__syncthreads();

	df[globalIdx] +=
		( c_ax * ( s_f[sj+si+1] - s_f[sj+si-1] )
		+ c_bx * ( s_f[sj+si+2] - s_f[sj+si-2] )
		+ c_cx * ( s_f[sj+si+3] - s_f[sj+si-3] )
		+ c_dx * ( s_f[sj+si+4] - s_f[sj+si-4] ) );
}


__global__ void derivative_x_fir(float *f, float *df)
{
	extern __shared__ float s_f[]; // 4-wide halo

	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int k  = blockIdx.y;
	int si = i + 4;       // local i for shared memory access + halo offset
	int sj = (c_mx+8)*threadIdx.y; // local j for shared memory access

	int globalIdx = k * c_mx * my + j * c_mx + i;

	s_f[sj+si] = f[globalIdx];

	__syncthreads();

	// fill in periodic images in shared memory array
	if (i < 4)
	{
		s_f[sj+si-4]  = s_f[sj+si+c_mx-5];
		s_f[sj+si+c_mx] = s_f[sj+si+1];
	}

	__syncthreads();

	// the taylor series expansion has been reformulated to look like a FIR filter
	float* z = &s_f[sj+si-4];
	float* c = c_coef_x;
	int count = 9;
	
	while(count--)
		 df[globalIdx] += (*c++)*(*z++);
	
}

// -------------
// y derivatives
// -------------
__global__ void derivative_y_fixed(float *f, float *df)
{
	__shared__ float s_f[my+8][sPencils];

	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = threadIdx.y;
	int k  = blockIdx.y;
	int si = threadIdx.x;
	int sj = j + 4;

	int globalIdx = k * mx * my + j * mx + i;

	s_f[sj][si] = f[globalIdx];

	__syncthreads();

	if (j < 4)
	{
		s_f[sj-4][si]  = s_f[sj+my-5][si];
		s_f[sj+my][si] = s_f[sj+1][si];
	}

	__syncthreads();

	df[globalIdx] =
		( c_ay * ( s_f[sj+1][si] - s_f[sj-1][si] )
		+ c_by * ( s_f[sj+2][si] - s_f[sj-2][si] )
		+ c_cy * ( s_f[sj+3][si] - s_f[sj-3][si] )
		+ c_dy * ( s_f[sj+4][si] - s_f[sj-4][si] ) );
}


__global__ void derivativeAccumY(float *f, float *df)
{
	extern __shared__ float s_f[]; // 4-wide halo

	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = threadIdx.y;
	int k  = blockIdx.y;
	int si = threadIdx.x;
	int sj = j + 4;

	int globalIdx = k * c_mx * c_my + j * c_mx + i;

	s_f[(sPencils)*sj+si] = f[globalIdx];

	__syncthreads();

	if (j < 4)
	{
		s_f[(sPencils)*(sj-4)+si]  = s_f[(sPencils)*(sj+c_my-5)+si];
		s_f[(sPencils)*(sj+c_my)+si] = s_f[(sPencils)*(sj+1)+si];
	}

	__syncthreads();

	df[globalIdx] +=
		( c_ay * ( s_f[(sPencils)*(sj+1)+si] - s_f[(sPencils)*(sj-1)+si] )
		+ c_by * ( s_f[(sPencils)*(sj+2)+si] - s_f[(sPencils)*(sj-2)+si] )
		+ c_cy * ( s_f[(sPencils)*(sj+3)+si] - s_f[(sPencils)*(sj-3)+si] )
		+ c_dy * ( s_f[(sPencils)*(sj+4)+si] - s_f[(sPencils)*(sj-4)+si] ) );
}


// ------------
// z derivative
// ------------
__global__ void derivative_z_fixed(float *f, float *df)
{
	__shared__ float s_f[mz+8][sPencils];

	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = blockIdx.y;
	int k  = threadIdx.y;
	int si = threadIdx.x;
	int sk = k + 4; // halo offset

	int globalIdx = k * mx * my + j * mx + i;

	s_f[sk][si] = f[globalIdx];

	__syncthreads();

	if (k < 4)
	{
		s_f[sk-4][si]  = s_f[sk+mz-5][si];
		s_f[sk+mz][si] = s_f[sk+1][si];
	}

	__syncthreads();

	df[globalIdx] =
		( c_az * ( s_f[sk+1][si] - s_f[sk-1][si] )
		+ c_bz * ( s_f[sk+2][si] - s_f[sk-2][si] )
		+ c_cz * ( s_f[sk+3][si] - s_f[sk-3][si] )
		+ c_dz * ( s_f[sk+4][si] - s_f[sk-4][si] ) );
}

__global__ void derivativeAccumZ(float *f, float *df)
{
//  __shared__ float s_f[mz+8][sPencils];
	extern __shared__ float s_f[]; // 4-wide halo

	int i  = blockIdx.x*blockDim.x + threadIdx.x;
	int j  = blockIdx.y;
	int k  = threadIdx.y;
	int si = threadIdx.x;
	int sk = k + 4; // halo offset

	int globalIdx = k * mx * my + j * mx + i;

	s_f[(sPencils)*sk+si] = f[globalIdx];

	__syncthreads();

	if (k < 4)
	{
		s_f[(sPencils)*(sk-4)+si]  = s_f[(sPencils)*(sk+mz-5)+si];
		s_f[(sPencils)*(sk+mz)+si] = s_f[(sPencils)*(sk+1)+si];
	}

	__syncthreads();

	df[globalIdx] +=
		( c_az * ( s_f[(sPencils)*(sk+1)+si] - s_f[(sPencils)*(sk-1)+si] )
		+ c_bz * ( s_f[(sPencils)*(sk+2)+si] - s_f[(sPencils)*(sk-2)+si] )
		+ c_cz * ( s_f[(sPencils)*(sk+3)+si] - s_f[(sPencils)*(sk-3)+si] )
		+ c_dz * ( s_f[(sPencils)*(sk+4)+si] - s_f[(sPencils)*(sk-4)+si] ) );
}




// Run the kernels for a given dimension. One for sPencils, one for lPencils
extern "C" void runTest(int dimension)
{
	void (*fpDeriv[2])(float*, float*);

	switch(dimension) {
	case 0:
		fpDeriv[0] = derivativeAccumX;
		fpDeriv[1] = derivative_x_fir;
	break;
	case 1:
		fpDeriv[0] = derivativeAccumY;
		fpDeriv[1] = derivative_y_fixed;
	break;
	case 2:
		fpDeriv[0] = derivativeAccumZ;
		fpDeriv[1] = derivative_z_fixed;
	break;
	}

	int sharedDims[3][2][2] = {
		mx, sPencils,
		mx, sPencils,

		sPencils, my,
		sPencils, my,

		sPencils, mz,
		sPencils, mz };

	float *f = new float[mx*my*mz];
	float *df = new float[mx*my*mz];
	float *sol = new float[mx*my*mz];

	initInput(f, dimension);
	initSol(sol, dimension);

	// device arrays
	int bytes = mx*my*mz * sizeof(float);
	float *d_f, *d_df;

	checkCuda( cudaMalloc((void**)&d_f, bytes) );
	checkCuda( cudaMalloc((void**)&d_df, bytes) );

	const int nReps = 20;
	float milliseconds;
	cudaEvent_t startEvent, stopEvent;

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	double error, maxError;

	printf("%c derivatives\n\n", (char)(0x58 + dimension));

	for (int fp = 0; fp < 2; fp++)
	{
		checkCuda( cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice) );

		fpDeriv[fp]<<<numBlocks[dimension][fp],threadsPerBlock[dimension][fp],0x2000>>>(d_f, d_df); // warm up
		checkCuda( cudaEventRecord(startEvent, 0) );
		for (int i = 0; i < nReps; i++)
		{
			checkCuda( cudaMemset(d_df, 0, bytes) );

			fpDeriv[fp]<<<numBlocks[dimension][fp],threadsPerBlock[dimension][fp],0x2000>>>(d_f, d_df);
		}

		checkCuda( cudaEventRecord(stopEvent, 0) );
		checkCuda( cudaEventSynchronize(stopEvent) );
		checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

		checkCuda( cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost) );

		checkResults(error, maxError, sol, df);

		printf("  Using shared memory tile of %d x %d\n",
			sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
		printf("   RMS error: %e\n", error);
		printf("   MAX error: %e\n", maxError);
		printf("   Average time (ms): %f\n", milliseconds / nReps);
		printf("   Average Bandwidth (GB/s): %f\n\n",
		   2.f * 1e-6 * mx * my * mz * nReps * sizeof(float) / milliseconds);
	}

	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );

	checkCuda( cudaFree(d_f) );
	checkCuda( cudaFree(d_df) );

	delete [] f;
	delete [] df;
	delete [] sol;
}

// this work is based on
//Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
struct simulation_space
{
	dim3 size;	// the size of the simulation space
	float* d_dField;	// electric flux density, see Sullivan P.32 chapter 2.1
	float* d_eField;	// the electrical field
	float* d_hField;	// the magnetic field
	float* d_i;		// a parameter that stores a current (efield* conductivity) like parameter
	float* d_ga;		// relative permittivity (with some time varying things)
	float* d_gb;		// the conductivity (some time varient 	stuff)
};


// fixme wrap these things into a structure
struct simulation_space simSpaceX;	// x component fields
struct simulation_space simSpaceY;	// y component fields
struct simulation_space simSpaceZ;	// z component fields
float* cpuWorkingSpace;		// this is a space the same size as the volume as we work with in the GPU, use it as a temporary work space

template <typename T>
__global__ void arraySet(int n, T* ptr, T val)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		ptr[i] = val;
}


extern int SimulationSpace_Reset( struct simulation_space* pSpace)
{
	int retval = 0;
	int bytes = pSpace->size.x * pSpace->size.y * pSpace->size.x * sizeof(float);
	retval += checkCuda( cudaMemset(pSpace->d_eField, 0, bytes) );
	retval += checkCuda( cudaMemset(pSpace->d_dField, 0, bytes) );
	retval += checkCuda( cudaMemset(pSpace->d_hField, 0, bytes) );
	retval += checkCuda( cudaMemset(pSpace->d_i, 0, bytes) );

	int blockSize = 256;
	int numBlocks = ((bytes/sizeof(float)) + blockSize - 1) / blockSize;
	arraySet<<<numBlocks, blockSize>>>(bytes/sizeof(float), pSpace->d_ga, (float)1.0);

	retval += checkCuda( cudaMemset(pSpace->d_gb, 0, bytes) );
	return(retval);
}

// initialze simulation space to all zeros
extern int SimulationSpace_ResetFields(void)
{
	int retval = 0;
	retval += SimulationSpace_Reset(&simSpaceX);
	retval += SimulationSpace_Reset(&simSpaceY);
	retval += SimulationSpace_Reset(&simSpaceZ);
	return(retval);
}


// allocates storage on the GPU to store the simulation state information,
// based on the size of the supplied geometry
extern int SimulationSpace_CreateDim(dim3* sim_size, struct simulation_space* pSpace)
{
	int retval = 0;
	pSpace->size.x = sim_size->x;
	pSpace->size.y = sim_size->y;
	pSpace->size.z = sim_size->z;
	int bytes = pSpace->size.x * pSpace->size.y * pSpace->size.x * sizeof(float);
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_dField, bytes) );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_eField, bytes) );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_hField, bytes) );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_i, bytes) );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_ga, bytes) );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_gb, bytes) );

	return(retval);
}


extern int SimulationSpace_DestroyDim(struct simulation_space* pSpace)
{
	int retval = 0;
	retval += checkCuda( cudaFree(pSpace->d_dField) );
	retval += checkCuda( cudaFree(pSpace->d_eField) );
	retval += checkCuda( cudaFree(pSpace->d_hField) );
	retval += checkCuda( cudaFree(pSpace->d_i) );
	retval += checkCuda( cudaFree(pSpace->d_ga) );
	retval += checkCuda( cudaFree(pSpace->d_gb) );
	return(retval);
}



// allocates storage on the GPU to store the simulation state information,
// based on the size of the supplied geometry
extern int SimulationSpace_Create(dim3* sim_size)
{
	int retval = 0;
	int bytes = sim_size->x * sim_size->y * sim_size->x * sizeof(float);

	cpuWorkingSpace = (float*)malloc(bytes);
printf("%s allocating %d(kB) (%d, %d, %d)\n",__FUNCTION__, 6*3*bytes/1024, sim_size->x,sim_size->y,sim_size->z);
	retval += SimulationSpace_CreateDim(sim_size, &simSpaceX);
	retval += SimulationSpace_CreateDim(sim_size, &simSpaceY);
	retval += SimulationSpace_CreateDim(sim_size, &simSpaceZ);
printf("spaces allocated, initializing\n");
	retval += SimulationSpace_ResetFields();
printf("initialized\n");

	return(retval);
}


extern int SimulationSpace_Destroy(void)
{
	int retval = 0;
	retval += SimulationSpace_DestroyDim(&simSpaceX);
	retval += SimulationSpace_DestroyDim(&simSpaceY);
	retval += SimulationSpace_DestroyDim(&simSpaceZ);

	free(cpuWorkingSpace);

	return(retval);
}


