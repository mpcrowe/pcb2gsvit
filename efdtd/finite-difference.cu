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
// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
const int sPencils = 4;  // small # pencils

__constant__ int c_mx, c_my, c_mz;	// size of the simulation space instance in gpu space
__constant__ int c_numElements;
__constant__ float c_delExp;  // coeff used in calculating the efield
__constant__ float* c_ga;
__constant__ float* c_gb;
__constant__ float* c_gc;
__constant__ char* c_mi;

// stencil coefficients, used in computing the partial derivative
__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;
__constant__ float c_coef_x[9];


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
	struct vector_field sField;	// ????
	char* d_mat_index;	//
	float* d_ga;		// relative permittivity (with some time varying things)
	float* d_gb;		// the conductivity (some time varient 	stuff)
	float* d_gc;		// the frequency dependant media value
	float dx;
	float dy;
	float dz;
};


// fixme wrap these things into a structure
int spaceInitialized = 0;
struct simulation_space simSpace;	// component fields
//float* cpuWorkingSpace;		// this is a space the same size as the volume as we work with in the GPU, use it as a temporary work space



static int updateDerivativeParametersX(int voxels, float scale)
{
	float delta = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len x' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float ax =  4.f / 5.f   * delta;
	float bx = -1.f / 5.f   * delta;
	float cx =  4.f / 105.f * delta;
	float dx = -1.f / 280.f * delta;
	checkCuda( cudaMemcpyToSymbol(c_ax, &ax, sizeof(float)), __LINE__ );
	checkCuda( cudaMemcpyToSymbol(c_bx, &bx, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cx, &cx, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dx, &dx, sizeof(float)), __LINE__  );

	float coef_x[9];
	coef_x[0] = (1.f  / 280.f) * delta;	// -dx
	coef_x[1] = (-4.f / 105.f) * delta;	// -cx
	coef_x[2] = (1.f  / 5.f)   * delta;	// -bx
	coef_x[3] = (-4.f / 5.f)   * delta;	// -ax
	coef_x[4] = 0.f;
	coef_x[5] = (4.f  / 5.f)   * delta;	// ax
	coef_x[6] = (-1.f / 5.f)   * delta;	// bx
	coef_x[7] = (4.f  / 105.f) * delta;	// cx
	coef_x[8] = (-1.f / 280.f) * delta;	// dx
	checkCuda( cudaMemcpyToSymbol(c_coef_x, coef_x, 9* sizeof(float)), __LINE__  );

	return (0);
}


static int updateDerivativeParametersY(int voxels, float scale)
{
	float delta = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len y' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float ay =  4.f / 5.f   * delta;
	float by = -1.f / 5.f   * delta;
	float cy =  4.f / 105.f * delta;
	float dy = -1.f / 280.f * delta;
	checkCuda( cudaMemcpyToSymbol(c_ay, &ay, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_by, &by, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cy, &cy, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dy, &dy, sizeof(float)), __LINE__  );


	return (0);
}


static int updateDerivativeParametersZ(int voxels, float scale)
{
	float delta = (voxels-1.f)*scale;
	if((voxels % sPencils != 0) )
	{
                printf("'len Z' must be integral multiples of sPencils %d, %d\n", voxels, sPencils);
                exit(1);
        }

	float az =  4.f / 5.f   * delta;
	float bz = -1.f / 5.f   * delta;
	float cz =  4.f / 105.f * delta;
	float dz = -1.f / 280.f * delta;
	checkCuda( cudaMemcpyToSymbol(c_az, &az,sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_bz, &bz, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_cz, &cz, sizeof(float)), __LINE__  );
	checkCuda( cudaMemcpyToSymbol(c_dz, &dz, sizeof(float)), __LINE__  );

	return (0);
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


static void partialX(float* dest, float* src, float scale, dim3 size)
{
	dim3 nBlocks_x  = dim3(size.y / sPencils, size.z, 1);
	dim3 nThreads_x = dim3(size.x, sPencils, 1);

	updateDerivativeParametersX(size.x, scale);
	derivativeAccumX<<<nBlocks_x,nThreads_x,SHARED_SIZE>>>(dest,src);
}


static void partialY(float* dest, float* src, float scale, dim3 size)
{
	dim3 nBlocks_y = dim3(size.x / sPencils, size.z, 1);
        dim3 nThreads_y = dim3(sPencils, size.y, 1);

	updateDerivativeParametersY(size.y, scale);
	derivativeAccumY<<<nBlocks_y,nThreads_y,SHARED_SIZE>>>(dest,src);
}


static void partialZ(float* dest, float* src, float scale, dim3 size)
{
        dim3 nBlocks_z  = dim3(size.x / sPencils, size.y, 1);
        dim3 nThreads_z = dim3(sPencils, size.z, 1);

	updateDerivativeParametersZ(size.z, scale);
	derivativeAccumZ<<<nBlocks_z,nThreads_z,SHARED_SIZE>>>(dest,src);
}


// D(t+1) = D(t) + curl(H)
// this is used to compute the flux density and the magnetic field
static void curlAccum(struct vector_field* dest, struct vector_field* src, float scale, dim3 size)
{
//printf("%s\n", __FUNCTION__);
	//Dx(t+1) = Dx(t)+ (dHz/dy - dHy/dz)
    	// dHz/dy
	partialY(dest->d_x, src->d_z, scale/simSpace.dy, size);
    	// -dHy/dz
	partialZ(dest->d_x, src->d_y, -scale/simSpace.dz, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

	//Dy(t+1) = Dy(t)+ (dHx/dz - dHz/dx)
    	// dHx/dz
	partialZ(dest->d_y, src->d_x, scale/simSpace.dz, size);
    	// -dHz/dx
	partialX(dest->d_y, src->d_z, -scale/simSpace.dx, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

	//Dz(t+1) = Dz(t)+ (dHy/dx - dHx/dy)
    	// dHy/dx
	partialX(dest->d_z, src->d_y, scale/simSpace.dx, size);
    	// -dHx/dy
	partialY(dest->d_z, src->d_x, -scale/simSpace.dy, size);
	checkCuda(cudaDeviceSynchronize(), __LINE__);
}


// D(t+1) = D(t) + curl(H)
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
// p.80-81
void fluxDensity_step(void)
{
	//Dx(t+1) = Dx(t)+ (dHz/dy - dHy/dz)
//printf("%s\n", __FUNCTION__);
	curlAccum(&simSpace.dField, &simSpace.hField, 1.0f, simSpace.size);
}


// H(t+1) = H(t) - curl(E)
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
// p.80-81
void magneticField_step(void)
{
	//Hx(t+1) = Hx(t)+ (dEy/dz - dEz/dy)
//printf("%s\n", __FUNCTION__);
	curlAccum(&simSpace.hField, &simSpace.eField, -1.0f, simSpace.size);
}


template <typename T>
__global__ void iFieldDir_step(T* d_i, T* d_e )
{
	// i = i + gb*e
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < c_numElements; i += stride)
	{
		char materialIndex = c_mi[i];
		T gb = c_gb[materialIndex];
		d_i[i] += gb * d_e[i];
	}
}

static int currentField_step(void)
{
	int retval = 0;
	int bytes = simSpace.size.x * simSpace.size.y * simSpace.size.z * sizeof(float);
        int blockSize = 256;
        int numBlocks = ((bytes/sizeof(float)) + blockSize - 1) / blockSize;
//printf("%s numBlocks%d, blockSize:%d\n", __FUNCTION__, numBlocks, blockSize);

	// compute field for each axis

	// Ix
       	iFieldDir_step<<<numBlocks, blockSize>>>( simSpace.iField.d_x, simSpace.eField.d_x);

	// Iy
       	iFieldDir_step<<<numBlocks, blockSize>>>( simSpace.iField.d_y, simSpace.eField.d_y);

	// Iz
       	iFieldDir_step<<<numBlocks, blockSize>>>( simSpace.iField.d_z, simSpace.eField.d_z);

	return(retval);
}



template <typename T>
__global__ void eFieldDir_step(T* d_e, T* d_d, T* d_i, T* d_s )
{
	// e = ga * (d -i - del_exp* s)
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < c_numElements; i += stride)
	{
		char materialIndex = c_mi[i];
		if(materialIndex>127)
			materialIndex = 127;
		if(materialIndex<0)
			materialIndex = 0;
		T ga = c_ga[materialIndex];
		d_e[i] = ga*(d_d[i] - d_i[i] - c_delExp * d_s[i]) ;
	}
}

float local_del_exp = 0.0f;
static int electricField_step(void)
{
	int retval = 0;
	int bytes = simSpace.size.x * simSpace.size.y * simSpace.size.z * sizeof(float);
        int blockSize = 256;
        int numBlocks = ((bytes/sizeof(float)) + blockSize - 1) / blockSize;
//printf("%s numBlocks%d, blockSize:%d\n", __FUNCTION__, numBlocks, blockSize);

	// fixme these remain constant through out simulation, move to a
	// better spot in initialization
	retval += checkCuda( cudaMemcpyToSymbol(c_delExp, &local_del_exp, sizeof(float)), __LINE__  );
	if(retval)
		return(retval);

	// compute field for each axis

	// Ex
       	eFieldDir_step<<<numBlocks, blockSize>>>( simSpace.eField.d_x, simSpace.dField.d_x, simSpace.iField.d_x, simSpace.sField.d_x);
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	// Ey
       	eFieldDir_step<<<numBlocks, blockSize>>>( simSpace.eField.d_y, simSpace.dField.d_y, simSpace.iField.d_y, simSpace.sField.d_y);
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	// Ez
       	eFieldDir_step<<<numBlocks, blockSize>>>( simSpace.eField.d_z, simSpace.dField.d_z, simSpace.iField.d_z, simSpace.sField.d_z);
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	return(retval);
}


template <typename T>
__global__ void arraySet(int n, T* ptr, T val)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		ptr[i] = val;
}


template <typename T>
__global__ void insert(T* dest, dim3 dimDest, T* src, dim3 dimSrc, dim3 offset, T maskVal)
{
	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int srcIndex = i*blockDim.y + j;
	int sz;

	for(sz=0;sz<dimSrc.z;sz++)
	{
		T val = src[(srcIndex*dimSrc.z)+sz];
		if(val !=maskVal)
		{
			int globalIdx = dimDest.y*dimDest.z*(i+offset.x) + dimDest.z*(j+offset.y ) + offset.z +sz;
			T* ptr = &dest[globalIdx];
			*ptr = val;
		}
	}
}


template <typename T>
static int _FillArb(T* dest, dim3 dimDest, char* src, dim3 dimSrc, int xCenter, int yCenter, int zStart, T maskVal)
{
	int retval = 0;
	char* d_src;
//	int numBytes = dimSrc.x * dimSrc.y * sizeof(char);
	if(zStart>dimDest.z)
		return(-1);


	// compute offset from dim and center
	//compute number of blocks and threads to cover space
	dim3 offset;
	offset.x = xCenter-dimSrc.x/2;
	offset.y = yCenter-dimSrc.y/2;
	offset.z = zStart;

	dim3 blockSize(dimSrc.x, dimSrc.y);
        dim3 numBlocks(1,1);

//	if(zStart+dimSrc.z> dimDest.z) 
//		zLen = dimDest.z-zStart;

        insert<<<numBlocks, blockSize>>>(dest, dimDest, d_src, dimSrc, offset, maskVal);

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	return(retval);
}


template <typename T>
__global__ void extrudeZ(T* dest, dim3 destSize, T* src, dim3 offset, int zLen, T maskVal)
{
	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int srcIndex = i*blockDim.y + j;

	T val = src[srcIndex];
	if(val !=maskVal)
	{
		int count = zLen;
		int globalIdx = destSize.y * destSize.z * (i+offset.x) + destSize.z * (j+offset.y ) + offset.z;
		T* ptr = &dest[globalIdx];
		while(count--)
		{
			*ptr++ = val;
		}
	}
}


template <typename T>
static int _ExtrudeZArb(T* dest, dim3 dSize, T* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen, T maskVal)
{
	int retval = 0;
	T* d_src;
	int numBytes = xDim * yDim * sizeof(T);
	if(zStart>dSize.z)
		return(-1);

	// move src into GPU space
        retval += checkCuda( cudaMalloc((void**)&d_src, numBytes), __LINE__  );
	retval += checkCuda( cudaMemcpy( d_src, src, numBytes, cudaMemcpyHostToDevice), __LINE__);


	// compute offset from dim and center
	//compute number of blocks and threads to cover space
	dim3 offset;
	offset.x = xCenter-xDim/2;
	offset.y = yCenter-yDim/2;
	offset.z = zStart;

	dim3 blockSize(xDim, yDim);
        dim3 numBlocks(1,1);

	if(zStart+zLen> dSize.z) 
		zLen = dSize.z-zStart;

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

        extrudeZ<<<numBlocks, blockSize>>>(dest, dSize, d_src, offset, zLen, maskVal);

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	// cleanup
	retval += checkCuda( cudaFree(d_src), __LINE__  );
  
	return(retval);
}


extern int SimulationSpace_ExtrudeZ(char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen)
{
	return(_ExtrudeZArb(simSpace.d_mat_index, simSpace.size, src, xDim, yDim, xCenter, yCenter, zStart, zLen, (char)0));
}

extern int MatIndex_ExtrudeZ(char* dest, int dx, int dy, int dz, char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen)
{
	dim3 size;
	size.x = dx;
	size.y = dy;
	size.z = dz;

	return(_ExtrudeZArb(dest, size, src, xDim, yDim, xCenter, yCenter, zStart, zLen,(char)0));
}


template <typename T>
__global__ void extrudeY(T* dest, dim3 destSize, T* src, dim3 offset, int yLen, T maskVal)
{
	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int srcIndex = i*blockDim.y + j;

	T val = src[srcIndex];
	if(val !=maskVal)
	{
		int count = yLen;
		int x = i+offset.x;
		int y = offset.y;
		int z = j+offset.z;
		int globalIdx = destSize.y*destSize.z*x + destSize.z*y + z;
		T* ptr = &dest[globalIdx];
		while(count--)
		{
			*ptr = val;
			ptr += destSize.z;
		}
	}
}


template <typename T>
static int _ExtrudeYArb(T* dest, dim3 dSize, char* src, int xDim, int zDim, int xCenter, int zCenter, int yStart, int yLen)
{
	int retval = 0;
	char* d_src;
	int numBytes = xDim * zDim * sizeof(char);
	if(yStart>dSize.y)
		return(-1);

	// move src into GPU space
        retval += checkCuda( cudaMalloc((void**)&d_src, numBytes), __LINE__  );
	retval += checkCuda( cudaMemcpy( d_src, src, numBytes, cudaMemcpyHostToDevice), __LINE__);


	// compute offset from dim and center
	//compute number of blocks and threads to cover space
	dim3 offset;
	offset.x = xCenter-xDim/2;
	offset.z = zCenter-zDim/2;
	offset.y = yStart;

	dim3 blockSize(xDim, zDim);
        dim3 numBlocks(1,1);

	if(yStart+yLen> dSize.y) 
		yLen = dSize.y-yStart;

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

        extrudeY<<<numBlocks, blockSize>>>(dest, dSize, d_src, offset, yLen, (char)0);

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	// cleanup
	retval += checkCuda( cudaFree(d_src), __LINE__  );
  
	return(retval);
}

extern int SimulationSpace_ExtrudeY(char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen)
{
	return(_ExtrudeYArb(simSpace.d_mat_index, simSpace.size,
		src, xDim, yDim, xCenter, yCenter, zStart, zLen));
}


template <typename T>
__global__ void extrudeX(T* dest, dim3 destSize, T* src, dim3 offset, int xLen, T maskVal)
{
	int i   = threadIdx.x;
	int j   = blockIdx.x*blockDim.y + threadIdx.y;
	int srcIndex = i*blockDim.y + j;

	T val = src[srcIndex];
	if(val !=maskVal)
	{
		int count = xLen;
		int x = offset.x;
		int y = i+offset.y;
		int z = j+offset.z;
		int globalIdx = destSize.y*destSize.z*x + destSize.z*y + z;
		T* ptr = &dest[globalIdx];
		while(count--)
		{
			*ptr = val;
			ptr += destSize.y * destSize.z;
		}
	}
}

template <typename T>
static int _ExtrudeXArb(T* dest, dim3 dSize, char* src, int yDim, int zDim, int yCenter, int zCenter, int xStart, int xLen)
{
	int retval = 0;
	char* d_src;
	int numBytes = yDim * zDim * sizeof(char);
	if(xStart>dSize.x)
		return(-1);

	// move src into GPU space
        retval += checkCuda( cudaMalloc((void**)&d_src, numBytes), __LINE__  );
	retval += checkCuda( cudaMemcpy( d_src, src, numBytes, cudaMemcpyHostToDevice), __LINE__);


	// compute offset from dim and center
	//compute number of blocks and threads to cover space
	dim3 offset;
	offset.y = yCenter-yDim/2;
	offset.z = zCenter-zDim/2;
	offset.x = xStart;

	dim3 blockSize(yDim, zDim);
        dim3 numBlocks(1,1);
	if(xStart+xLen> dSize.x) 
		xLen = dSize.x-xStart;

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

        extrudeX<<<numBlocks, blockSize>>>(dest, dSize, d_src, offset, xLen, (char)0);

	retval += checkCuda(cudaDeviceSynchronize(), __LINE__);
	if(retval)
		return(retval);

	// cleanup
	retval += checkCuda( cudaFree(d_src), __LINE__  );
  
	return(retval);
}


extern int SimulationSpace_ExtrudeX(char* src, int xDim, int yDim, int xCenter, int yCenter, int zStart, int zLen)
{
	return(_ExtrudeXArb(simSpace.d_mat_index, simSpace.size,
		src, xDim, yDim, xCenter, yCenter, zStart, zLen));
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
	int numElmnts = pSpace->size.x * pSpace->size.y * pSpace->size.z;
	retval += VectorField_Zero(&pSpace->eField, pSpace->size );
	retval += VectorField_Zero(&pSpace->dField, pSpace->size );
	retval += VectorField_Zero(&pSpace->hField, pSpace->size );

	retval += VectorField_Zero(&pSpace->iField, pSpace->size );
	retval += VectorField_Zero(&pSpace->sField, pSpace->size );


	int blockSize = 256;
	int numBlocks = ((numElmnts) + blockSize - 1) / blockSize;
	arraySet<<<numBlocks, blockSize>>>(MAX_SIZE_MATERIAL_TABLE, pSpace->d_ga, (float)1.0);

	retval += checkCuda( cudaMemset(pSpace->d_gb, 0, MAX_SIZE_MATERIAL_TABLE * sizeof(float)), __LINE__  ); // bytes
	retval += checkCuda( cudaMemset(pSpace->d_gc, 0, MAX_SIZE_MATERIAL_TABLE * sizeof(float)), __LINE__  ); // bytes
	retval += checkCuda( cudaMemset(pSpace->d_mat_index, 0, numElmnts*sizeof(char)) , __LINE__  ); // bytes

	pSpace->dx = 1.0f;
	pSpace->dy = 1.0f;
	pSpace->dz = 1.0f;

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
	int numE = pSpace->size.x * pSpace->size.y * pSpace->size.z;
	retval += checkCuda( cudaMemcpyToSymbol(c_mx, &sim_size->x, sizeof(int)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_my, &sim_size->y, sizeof(int)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_mz, &sim_size->z, sizeof(int)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_numElements, &numE, sizeof(int)), __LINE__  );

	retval += VectorField_Malloc(&pSpace->dField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->eField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->hField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->iField, pSpace->size);
	retval += VectorField_Malloc(&pSpace->sField, pSpace->size);


	retval += checkCuda( cudaMalloc((void**)&pSpace->d_mat_index, numE * sizeof(char))  , __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_ga, MAX_SIZE_MATERIAL_TABLE * sizeof(float)), __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_gb, MAX_SIZE_MATERIAL_TABLE * sizeof(float)), __LINE__  );
	retval += checkCuda( cudaMalloc((void**)&pSpace->d_gc, MAX_SIZE_MATERIAL_TABLE * sizeof(float)), __LINE__  );
	// put a copy of the pointers into const memory so that the GPU
	// functions can directly access these spaces
	retval += checkCuda( cudaMemcpyToSymbol(c_ga, &simSpace.d_ga, sizeof(float*)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_gb, &simSpace.d_gb, sizeof(float*)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_gc, &simSpace.d_gc, sizeof(float*)), __LINE__  );
	retval += checkCuda( cudaMemcpyToSymbol(c_mi, &simSpace.d_mat_index, sizeof(char*)), __LINE__  );


	return(retval);
}


static int SimulationSpace_DestroyDim(struct simulation_space* pSpace)
{
	int retval = 0;
	retval += VectorField_Free(&pSpace->dField);
	retval += VectorField_Free(&pSpace->eField);
	retval += VectorField_Free(&pSpace->hField);
	retval += VectorField_Free(&pSpace->iField);
	retval += VectorField_Free(&pSpace->sField);
	retval += checkCuda( cudaFree(pSpace->d_mat_index), __LINE__  );
	retval += checkCuda( cudaFree(pSpace->d_ga), __LINE__  );
	retval += checkCuda( cudaFree(pSpace->d_gb), __LINE__  );
	retval += checkCuda( cudaFree(pSpace->d_gc), __LINE__  );
	return(retval);
}


extern void SimulationSpace_Timestep(void)
{
printf("%s\n", __FUNCTION__);
//	checkCuda(cudaDeviceSynchronize(), __LINE__);

	fluxDensity_step();
	electricField_step();
	currentField_step();
	// fixme calculate loss factor used  for freq dependant media
	// sField update needed here
	// sField = del_exp*sField + gc*e
	magneticField_step();
}



// allocates storage on the GPU to store the simulation state information,
// based on the size of the supplied geometry
extern int SimulationSpace_Create(dim3* sim_size)
{
	int retval = 0;
	int bytes = sim_size->x * sim_size->y * sim_size->x * sizeof(float);
	if(spaceInitialized != 0)
	{
		fprintf(stderr, "%s Warning space alreaded allocated as (%d, %d, %d)\n",__FUNCTION__,simSpace.size.x,simSpace.size.y,simSpace.size.z);
		return(retval);
	}

//	cpuWorkingSpace = (float*)malloc(bytes);
printf("%s allocating %d(kB) (%d, %d, %d)\n",__FUNCTION__, 6*3*bytes/1024, sim_size->x,sim_size->y,sim_size->z);
	retval += SimulationSpace_CreateDim(sim_size, &simSpace);
printf("spaces allocated, initializing\n");
	retval += SimulationSpace_ResetFields();
printf("initialized\n");
	spaceInitialized = 1;
	return(retval);
}


extern int SimulationSpace_Destroy(void)
{
	int retval = 0;
	retval += SimulationSpace_DestroyDim(&simSpace);

//	free(cpuWorkingSpace);

	return(retval);
}


// inserts a "string" of z-axis spatial data into the material index
// matrix at the specified x,y and z offsets
// this function is used to build the image in the GPU that will be
// simulated
extern int FD_zlineInsert(char* zline, int x, int y, int z, int len)
{
	int retval;
	int offset = simSpace.size.y*simSpace.size.z * x + simSpace.size.z * y + z;
	if( offset > (simSpace.size.x*simSpace.size.y*simSpace.size.z) )
		return(-2);	// if out of bounds return error

	retval = FD_UpdateMatIndex(zline, len, offset);
	return(retval);
}


extern int FD_UpdateMatIndex(char* src, int len, int offset)
{
	int retval;
	int num_bytes = len * sizeof(char);
	char* dest = simSpace.d_mat_index;
	if(dest == NULL)	// if space not initialized return error
		return(-1);

	dest+=offset;

	retval = checkCuda( cudaMemcpy( dest, src, num_bytes, cudaMemcpyHostToDevice), __LINE__ );
	// sync needed because src may be overwritten by CPU after the
	// cudaMemcpy call
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__); 
	return(retval);
}


extern int FD_UpdateGa(float* src, int len)
{
	int retval;
	int num_bytes = len;
	if(len>MAX_SIZE_MATERIAL_TABLE)
		num_bytes = MAX_SIZE_MATERIAL_TABLE;
	num_bytes *= sizeof(float);
	retval = checkCuda( cudaMemcpy( simSpace.d_ga, src, num_bytes, cudaMemcpyHostToDevice), __LINE__ );
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__); 
	return(retval);
}


extern int FD_UpdateGb(float* src, int len)
{
	int retval;
	int num_bytes = len;
	if(len>MAX_SIZE_MATERIAL_TABLE)
		num_bytes = MAX_SIZE_MATERIAL_TABLE;
	num_bytes *= sizeof(float);
	retval = checkCuda( cudaMemcpy( simSpace.d_gb, src, num_bytes, cudaMemcpyHostToDevice), __LINE__ );
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__); 
	return(retval);
}


extern int FD_UpdateGc(float* src, int len)
{
	int retval;
	int num_bytes = len;
	if(len>MAX_SIZE_MATERIAL_TABLE)
		num_bytes = MAX_SIZE_MATERIAL_TABLE;
	num_bytes *= sizeof(float);
	retval = checkCuda( cudaMemcpy( simSpace.d_gc, src, num_bytes, cudaMemcpyHostToDevice), __LINE__ );
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__); 
	return(retval);
}

extern int FD_UpdateDelExp(float del_exp)
{
	int retval;
	local_del_exp = del_exp;
	retval = checkCuda( cudaMemcpyToSymbol(c_delExp, &del_exp, sizeof(float)), __LINE__  );
	if(retval)
		return(retval);
	retval += checkCuda(cudaDeviceSynchronize(), __LINE__); 
	return(retval);
}


extern void FD_UpdateDeltas(float dx, float dy, float dz)
{
	simSpace.dx = dx;
	simSpace.dy = dy;
	simSpace.dz = dz;
}


extern int FD_Testbed(void* image, int sx, int sy, int sz)
{
	int retval = 0;
	int numElements = sx*sy*sz;
	int bytes = numElements *sizeof(float);
//	int blockSize = 256;
//        int numBlocks = ((numElements) + blockSize - 1) / blockSize;
//	float* d_image;
// allocate a space to copy test data into, and copy it in
//        retval += checkCuda( cudaMalloc((void**)&d_image, bytes), __LINE__  );
//	retval += checkCuda( cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice), __LINE__  );
printf("%s\n", __FUNCTION__);

	// do test

	dim3 d(sx, sy, sz);	
	retval += SimulationSpace_Create(&d);
//	retval += checkCuda( cudaMemcpy(simSpace.hField.d_x, image, bytes, cudaMemcpyHostToDevice), __LINE__  );
	retval += checkCuda( cudaMemcpy(simSpace.hField.d_y, image, bytes, cudaMemcpyHostToDevice), __LINE__  );
//	retval += checkCuda( cudaMemcpy(simSpace.hField.d_z, image, bytes, cudaMemcpyHostToDevice), __LINE__  );
//	retval += checkCuda( cudaMemcpy(simSpace.dField.d_x, image, bytes, cudaMemcpyHostToDevice), __LINE__  );
	
//        arraySet<<<numBlocks, blockSize>>>(numElements, d_image, (float)-4.0);
//	arraySet<<<numBlocks, blockSize>>>(numElements, simSpace.eField.d_x, (float)-4.0);
	checkCuda(cudaDeviceSynchronize(), __LINE__);

	SimulationSpace_Timestep();
	SimulationSpace_Timestep();
	SimulationSpace_Timestep();
//	retval+= electricField_step();

	// write it back out to view result using openGL tools
//	retval += checkCuda( cudaMemcpy(image, simSpace.eField.d_x, bytes, cudaMemcpyDeviceToHost), __LINE__  );
	retval += checkCuda( cudaMemcpy(image, simSpace.d_mat_index, numElements, cudaMemcpyDeviceToHost), __LINE__  );

//	retval += checkCuda( cudaMemcpy(image, d_image, bytes, cudaMemcpyDeviceToHost), __LINE__  );
//	retval += checkCuda( cudaFree(d_image), __LINE__  );

	SimulationSpace_Destroy();

	return(retval);
}
