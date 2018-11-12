

#include "gpu_translate.h"

#include <stdio.h>
#include <stdlib.h>
#include <glib.h>  // for  using GList
#include <math.h>
#include <string.h>

#include "../src/material.h"
#include "finite-difference.h"

// this work is based on
// Dennis M. Sullivan "EM Simulation Using the FDTD Method", 2000 IEEE
// see page 25 (equations 2.22 and 2.23) for one dim solution which effectivly
// extends to three dimensions (except for the curl of the E and H fields)


// d_ga --> 1/(Er + (sigma*dt/E0) + (chi*dt/t0) )
//      where
//              Er = relative dielectric constant (1-15)
//              sigma = conductivity (S/m)
//              dt = timestep size (sec)
//              E0 = Absolute dielectric constant
//              chi = a dissapation factor related to the lossyness of the
//                      dielectric
//              t0 = time constant representing the cut off frequency of the
//                      lossy dielectric (sec)
// d_gb --> sigma*dt/E0
// d_gbc --> chi*dt/t0
// del_exp = exp^(-dt/t0)
#define EPS_Z 8.8541878176e-12
// updates the material properties table
extern int GT_UpdateMaterialTable(float dt  )
{
	float tmp[MAX_SIZE_MATERIAL_TABLE];
	int i;
	int retval;

	// update ga table (permittivity)
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		tmp[i] = 1.0f;
	}
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		float er = MATRL_Er(i);
		if( er < 0)
			break;
		float cond = MATRL_Cond(i);
		float chi = MATRL_Chi(i);
		float t0 = MATRL_T0(i);

		float ga = 1.0/(er +(cond*dt/EPS_Z) + (chi*dt/t0) );
		tmp[i] = ga;
	}
	retval = FD_UpdateGa(tmp, MAX_SIZE_MATERIAL_TABLE);
	if(retval)
	{
		fprintf(stderr, "%s, line:%d\n", __FUNCTION__, __LINE__);
		return(retval);
	}

	// update gb table (conductivity)
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		tmp[i] = 0.0f;
	}
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		float cond = MATRL_Cond(i);
		if( cond < 0)
			break;

		cond *= dt/EPS_Z;
		tmp[i] = cond;
	}
	retval = FD_UpdateGb(tmp, MAX_SIZE_MATERIAL_TABLE);
	if(retval)
	{
		fprintf(stderr, "%s, line:%d\n", __FUNCTION__, __LINE__);
		return(retval);
	}

	// update gc table (frequency dependant media)
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		tmp[i] = 0.0f;
	}
	for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
	{
		float chi = MATRL_Chi(i);
		if( chi < 0)
			break;

		chi *= dt/MATRL_T0(i);
		tmp[i] = chi;
	}
	retval = FD_UpdateGc(tmp, MAX_SIZE_MATERIAL_TABLE);
	if(retval)
	{
		fprintf(stderr, "%s, line:%d\n", __FUNCTION__, __LINE__);
		return(retval);
	}
	
	retval = FD_UpdateDelExp(expf(-dt));
	if(retval)
	{
		fprintf(stderr, "%s, line:%d\n", __FUNCTION__, __LINE__);
		return(retval);
	}
	
	


	//extern float MATRL_Ur(int index);
	//extern float MATRL_Cond(int index);
	//extern float MATRL_Sus(int index);

    return(0);


}


extern void GT_MakeVia(int xCenter, int yCenter, int outerRadius, int innerRadius, int start, int end, char matIndex)
{
	int rowSize = outerRadius*2+1;
	int colSize = rowSize;	// circles are round, but ExtrudeZ doesn't care
	int size = rowSize*colSize*sizeof(char);
	char* pTemplate = (char*)malloc(size);
	memset(pTemplate,0,size);
	int r;
	for(r=innerRadius; r<outerRadius; r++)
	{
		int x;
		int xOff = r;
		int yOff = r;
		for(x=0; x<=r; x++)
		{
			int y = (int)(sqrt(r*r-x*x)+0.5);
			// compute for one quadrant, apply to four quadrants
			int index = (x+xOff)*rowSize + (y+yOff);
			pTemplate[index] = matIndex;
			index = (x+xOff)*rowSize + (-y+yOff);
			pTemplate[index] = matIndex;
			index = (-x+xOff)*rowSize + (y+yOff);
			pTemplate[index] = matIndex;
			index = (-x+xOff)*rowSize + (-y+yOff);
			pTemplate[index] = matIndex;
		}
	}
//	int xDim, int yDim, int xCenter, int yCenter, int zStart, int zEnd
	SimulationSpace_ExtrudeZ(pTemplate, rowSize, colSize, xCenter, yCenter, start, end );
	free(pTemplate);	
}
