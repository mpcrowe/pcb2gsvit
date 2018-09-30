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
inline cudaError_t checkCuda(cudaError_t result, int lineNum)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: (%s) %d at line %d\n", cudaGetErrorString(result), result, lineNum);
	//	assert(result == cudaSuccess);
		exit(-1);
	}
#endif
	return result;
}

#define SHARED_SIZE 0x4000
float fx = 6.0f, fy = 1.0f, fz = 1.0f;
const int mx = 256, my = 256, mz = 256;
__constant__ int c_mx, c_my, c_mz;

// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
const int sPencils = 4;  // small # pencils

dim3 numBlocks[3][2], threadsPerBlock[3][2];

// stencil coefficients
__constant__ float c_ax, c_bx, c_cx, c_dx;
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
	checkCuda( cudaMemcpyToSymbol(c_ax, &ax, sizeof(float)), __LINE__ );
	checkCuda( cudaMemcpyToSymbol(c_bx, &bx, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cx, &cx, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dx, &dx, sizeof(float)), __LINE__  );

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
	checkCuda( cudaMemcpyToSymbol(c_coef_x, coef_x, 9* sizeof(float)), __LINE__  );

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
	checkCuda( cudaMemcpyToSymbol(c_ay, &ay, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_by, &by, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cy, &cy, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dy, &dy, sizeof(float)), __LINE__  );


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
	checkCuda( cudaMemcpyToSymbol(c_az, &az,sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_bz, &bz, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cz, &cz, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dz, &dz, sizeof(float)), __LINE__  );

	return (0);
}


// host routine to set constant data
extern "C" void setDerivativeParameters()
{
	setDerivativeParametersX(mx, 1.0);
	setDerivativeParametersY(my, 1.0);
	setDerivativeParametersZ(mz, 1.0);
	checkCuda( cudaMemcpyToSymbol(c_mx, &mx, sizeof(int)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_my, &my, sizeof(int)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_mz, &mz, sizeof(int)), __LINE__  );

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

	int globalIdx = k * c_mx * c_my + j * c_mx + i;

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

	int globalIdx = k * c_mx * c_my + j * c_mx + i;

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
	{  // this wraps the derivative space into a circle,  probably not what is desired for a real world problem
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
	{  // this wraps the space into a circle (ends overlap),  probably not what is desired for a real world problem
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

	int globalIdx = k * c_mx * c_my + j * c_mx + i;

	s_f[(sPencils)*sk+si] = f[globalIdx];

	__syncthreads();

	if (k < 4)
	{
		s_f[(sPencils)*(sk-4)+si]  = s_f[(sPencils)*(sk+c_mz-5)+si];
		s_f[(sPencils)*(sk+c_mz)+si] = s_f[(sPencils)*(sk+1)+si];
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

	checkCuda( cudaMalloc((void**)&d_f, bytes), __LINE__  );
	checkCuda( cudaMalloc((void**)&d_df, bytes), __LINE__  );

	const int nReps = 20;
	float milliseconds;
	cudaEvent_t startEvent, stopEvent;

	checkCuda( cudaEventCreate(&startEvent), __LINE__  );
	checkCuda( cudaEventCreate(&stopEvent), __LINE__  );

	double error, maxError;

	printf("%c derivatives\n\n", (char)(0x58 + dimension));

	for (int fp = 0; fp < 2; fp++)
	{
		checkCuda( cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice), __LINE__  );

		fpDeriv[fp]<<<numBlocks[dimension][fp],threadsPerBlock[dimension][fp],SHARED_SIZE>>>(d_f, d_df); // warm up
		checkCuda( cudaEventRecord(startEvent, 0), __LINE__  );
		for (int i = 0; i < nReps; i++)
		{
			checkCuda( cudaMemset(d_df, 0, bytes), __LINE__  );

			fpDeriv[fp]<<<numBlocks[dimension][fp],threadsPerBlock[dimension][fp],SHARED_SIZE>>>(d_f, d_df);
		}

		checkCuda( cudaEventRecord(stopEvent, 0), __LINE__  );
		checkCuda( cudaEventSynchronize(stopEvent), __LINE__  );
		checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent), __LINE__  );

		checkCuda( cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost), __LINE__  );

		checkResults(error, maxError, sol, df);

		printf("  Using shared memory tile of %d x %d\n", sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
		printf("   RMS error: %e\n", error);
		printf("   MAX error: %e\n", maxError);
		printf("   Average time (ms): %f\n", milliseconds / nReps);
		printf("   Average Bandwidth (GB/s): %f\n\n",
		   2.f * 1e-6 * mx * my * mz * nReps * sizeof(float) / milliseconds);
	}

	checkCuda( cudaEventDestroy(startEvent), __LINE__  );
	checkCuda( cudaEventDestroy(stopEvent), __LINE__  );

	checkCuda( cudaFree(d_f), __LINE__  );
	checkCuda( cudaFree(d_df), __LINE__  );

	delete [] f;
	delete [] df;
	delete [] sol;
}



// this work is based on
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE

// this structure is important because the curl computes the field in all
// three directions, without a structure to maintain the relationships,
// the bookeeping becomes complicated
struct vector_field
{
	float* d_x;	// pointer to a "flat" 3-d space located on the GPU, one for each direction
	float* d_y;
	float* d_z;
};

struct simulation_space
{
	dim3 size;	// the size of the simulation space
	struct vector_field dField;	// electric flux density, see Sullivan P.32 chapter 2.1
	struct vector_field eField;	// the electrical field
	struct vector_field hField;	// the magnetic field
	struct vector_field iField;	// a parameter that stores a current (efield* conductivity) like parameter
	float* d_ga;		// relative permittivity (with some time varying things)
	float* d_gb;		// the conductivity (some time varient 	stuff)
};


// fixme wrap these things into a structure
struct simulation_space simSpace;	// component fields
float* cpuWorkingSpace;		// this is a space the same size as the volume as we work with in the GPU, use it as a temporary work space

static void partialX(float* dest, float* src, float scale, dim3 size)
{
	dim3 nBlocks_x  = dim3(size.y / sPencils, size.z, 1);
	dim3 nThreads_x = dim3(size.x, sPencils, 1);

	setDerivativeParametersX(size.x, scale);
	derivativeAccumX<<<nBlocks_x,nThreads_x,SHARED_SIZE>>>(dest,src);
}


static void partialY(float* dest, float* src, float scale, dim3 size)
{
	dim3 nBlocks_y = dim3(size.x / sPencils, size.z, 1);
        dim3 nThreads_y = dim3(sPencils, size.y, 1);

	setDerivativeParametersY(size.y, scale);
	derivativeAccumY<<<nBlocks_y,nThreads_y,SHARED_SIZE>>>(dest,src);
}


static void partialZ(float* dest, float* src, float scale, dim3 size)
{
        dim3 nBlocks_z  = dim3(size.x / sPencils, size.y, 1);
        dim3 nThreads_z = dim3(sPencils, size.z, 1);

	setDerivativeParametersZ(size.z, scale);
	derivativeAccumZ<<<nBlocks_z,nThreads_z,SHARED_SIZE>>>(dest,src);
}


// D(t+1) = D(t) + curl(H)
// this is used to compute the flux density and the magnetic field
static void curlAccum(struct vector_field* dest, struct vector_field* src, float scale, dim3 size)
{
printf("%s\n", __FUNCTION__);
	//Dx(t+1) = Dx(t)+ (dHz/dy - dHy/dz)
    	// dHz/dy
	partialY(dest->d_x, src->d_z, scale, size);
    	// -dHy/dz
	partialZ(dest->d_x, src->d_y, -scale, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

	//Dy(t+1) = Dy(t)+ (dHx/dz - dHz/dx)
    	// dHx/dz
	partialZ(dest->d_y, src->d_x, scale, size);
    	// -dHz/dx
	partialX(dest->d_y, src->d_z, -scale, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

	//Dz(t+1) = Dz(t)+ (dHy/dx - dHx/dy)
    	// dHy/dx
	partialX(dest->d_z, src->d_y, scale, size);
    	// -dHx/dy
	partialY(dest->d_z, src->d_x, -scale, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

}


// D(t+1) = D(t) + curl(H)
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
// p.80-81
void fluxDensity_step(void)
{
	//Dx(t+1) = Dx(t)+ (dHz/dy - dHy/dz)
printf("%s\n", __FUNCTION__);
	curlAccum(&simSpace.dField, &simSpace.hField, 1.0f, simSpace.size);
}


// H(t+1) = H(t) - curl(E)
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
// p.80-81
void magneticField_step(void)
{
	//Hx(t+1) = Hx(t)+ (dEy/dz - dEz/dy)
printf("%s\n", __FUNCTION__);
	curlAccum(&simSpace.hField, &simSpace.eField, -1.0f, simSpace.size);
}

static void eFieldDir_step(float* d_e, float* d_d, float* d_i)
{
	// e = gax * (d -i - del_exp* s)
}

static void electricField_step(void)
{
	simSpace.eField.d_x, simSpace.dField.d_y, simSpace.iField,d_y 
}

template <typename T>
__global__ void arraySet(int n, T* ptr, T val)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		ptr[i] = val;
}


static int VectorField_Zero(struct vector_field* field, dim3 size)
{
	int retval = 0;
	int bytes = size.x * size.y * size.z * sizeof(float);
	retval += checkCuda( cudaMemset(field->d_x, 0, bytes), __LINE__  );
	retval += checkCuda( cudaMemset(field->d_y, 0, bytes), __LINE__  );
	retval += checkCuda( cudaMemset(field->d_z, 0, bytes), __LINE__  );
	return(retval);
}


static int VectorField_Free(struct vector_field* field)
{
	int retval = 0;
	retval += checkCuda( cudaFree(field->d_x), __LINE__  );
	retval += checkCuda( cudaFree(field->d_y), __LINE__  );
	retval += checkCuda( cudaFree(field->d_z), __LINE__  );
	return(retval);
}


static int VectorField_Malloc(struct vector_field* field, dim3 size)
{
	int retval = 0;
	int bytes = size.x * size.y * size.z * sizeof(float);
	retval += checkCuda( cudaMalloc((void**)&field->d_x, bytes), __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&field->d_y, bytes), __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&field->d_z, bytes), __LINE__  );

	return(retval);
}


static int SimulationSpace_Reset( struct simulation_space* pSpace)
{
	int retval = 0;
	int bytes = pSpace->size.x * pSpace->size.y * pSpace->size.z * sizeof(float);
	retval += VectorField_Zero(&pSpace->eField, pSpace->size );
	retval += VectorField_Zero(&pSpace->dField, pSpace->size );
	retval += VectorField_Zero(&pSpace->hField, pSpace->size );

	retval += VectorField_Zero(&pSpace->iField, pSpace->size );


	int blockSize = 256;
	int numBlocks = ((bytes/sizeof(float)) + blockSize - 1) / blockSize;
	arraySet<<<numBlocks, blockSize>>>(bytes/sizeof(float), pSpace->d_ga, (float)1.0);

	retval += checkCuda( cudaMemset(pSpace->d_gb, 0, bytes), __LINE__  );
	return(retval);
}


// initialze simulation space to all zeros
extern int SimulationSpace_ResetFields(void)
{
	int retval = 0;
	retval += SimulationSpace_Reset(&simSpace);
	return(retval);
}


// allocates storage on the GPU to store the simulation state information,
// based on the size of the supplied geometry
static int SimulationSpace_CreateDim(dim3* sim_size, struct simulation_space* pSpace)
{
	int retval = 0;
	pSpace->size.x = sim_size->x;
	pSpace->size.y = sim_size->y;
	pSpace->size.z = sim_size->z;
	retval += checkCuda( cudaMemcpyToSymbol(c_mx, &sim_size->x, sizeof(int)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_my, &sim_size->y, sizeof(int)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_mz, &sim_size->z, sizeof(int)), __LINE__  );

	retval += VectorField_Malloc(&pSpace->dField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->eField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->hField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->iField, pSpace->size);

	int bytes = pSpace->size.x * pSpace->size.y * pSpace->size.z * sizeof(float);
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_ga, bytes), __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_gb, bytes), __LINE__  );

	return(retval);
}


static int SimulationSpace_DestroyDim(struct simulation_space* pSpace)
{
	int retval = 0;
	retval += VectorField_Free(&pSpace->dField);
	retval += VectorField_Free(&pSpace->eField);
	retval += VectorField_Free(&pSpace->hField);
	retval += VectorField_Free(&pSpace->iField);
	retval += checkCuda( cudaFree(pSpace->d_ga), __LINE__  );
	retval += checkCuda( cudaFree(pSpace->d_gb), __LINE__  );
	return(retval);
}


extern void SimulationSpace_Timestep(void)
{
printf("%s\n", __FUNCTION__);
	fluxDensity_step();
	magneticField_step();
}



// allocates storage on the GPU to store the simulation state information,
// based on the size of the supplied geometry
extern int SimulationSpace_Create(dim3* sim_size)
{
	int retval = 0;
	int bytes = sim_size->x * sim_size->y * sim_size->x * sizeof(float);

	cpuWorkingSpace = (float*)malloc(bytes);
printf("%s allocating %d(kB) (%d, %d, %d)\n",__FUNCTION__, 6*3*bytes/1024, sim_size->x,sim_size->y,sim_size->z);
	retval += SimulationSpace_CreateDim(sim_size, &simSpace);
printf("spaces allocated, initializing\n");
	retval += SimulationSpace_ResetFields();
printf("initialized\n");

	return(retval);
}


extern int SimulationSpace_Destroy(void)
{
	int retval = 0;
	retval += SimulationSpace_DestroyDim(&simSpace);

	free(cpuWorkingSpace);

	return(retval);
}


