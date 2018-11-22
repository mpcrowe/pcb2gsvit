
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <stdlib.h>
#include <argp.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <glib.h>  // for  using GList
#include <string.h>

#include "../src/xpu.h"
#include "../src/xpathConsts.h"
#include "../src/material.h"


// This the main host code for the finite difference
// example.  The kernels are contained in the derivative_m module
#include "finite-difference.h"

static char* getXemFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return(XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_NAME) );
}

static char* getRiffFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return(XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_RIFF_FILENAME) );
}

extern  int FP_ReadRiff(char* riffFname)
{
	int retval;
	FILE* rifffd = fopen(riffFname, "r");
	if(rifffd == NULL)
	{
		fprintf(stderr, "Unable to open <%s>\n", riffFname);
		goto processingFault;
	}
	char chnkName[5];
	int32_t chnkSize;
printf("%s reading %s\n",__FUNCTION__, riffFname);
		
	while(1)
	{
		retval = fread(chnkName, sizeof(char), 4, rifffd);
		retval += fread(&chnkSize, sizeof(int32_t), 1, rifffd);
		if(retval != 5)
		{
			if(feof(rifffd))
				break;
			fprintf(stderr, "Header read error <%s> %d != %d\n", riffFname, retval,  5);
			goto processingFault;
		}
		chnkName[4] = 0;
		if( chnkName[0]=='f' && chnkName[1]=='d' && chnkName[2]=='t' && chnkName[3]=='d')
		{
			int32_t s_x = 0;
			int32_t s_y = 0;
			int32_t s_z = 0;
			int32_t off_x = 0;
			int32_t off_y = 0;
			int32_t off_z = 0;
			int32_t i, j;
		
			retval = fread(&s_x,sizeof(int32_t), 1, rifffd);
			retval += fread(&s_y,sizeof(int32_t), 1, rifffd);
			retval += fread(&s_z,sizeof(int32_t), 1, rifffd);
			retval += fread(&off_x,sizeof(int32_t), 1, rifffd);
			retval += fread(&off_y,sizeof(int32_t), 1, rifffd);
			retval += fread(&off_z,sizeof(int32_t), 1, rifffd);

			if(retval != 6)
			{
				fprintf(stderr, "Header read error <%s> %d != %d\n", riffFname, retval,  6);
				goto processingFault;
			}
			fprintf(stdout, "x:%d, y:%d z:%d  offx:%d offy:%d offz:%d chnk:%d\n", s_x, s_y, s_z, off_x, off_y, off_z, chnkSize);
			dim3 gpuSize;
			gpuSize.x = s_x+off_x;
			gpuSize.y = s_y+off_y;
			gpuSize.z = s_z+off_z;
			
			if(gpuSize.x%4 != 0)
				gpuSize.x = ((gpuSize.x/4)+1)*4;
			if(gpuSize.y%4 != 0)
				gpuSize.y = ((gpuSize.y/4)+1)*4;
			if(gpuSize.z%4 != 0)
				gpuSize.z = ((gpuSize.z/4)+1)*4;
			
			retval = SimulationSpace_Create(&gpuSize);
			if(retval)
				goto processingFault;
			
			char* zline = (char*)malloc(sizeof(char)*s_z);
			for(i=0; i<s_x; i++)
			{
				for(j=0; j<s_y; j++)
				{
					retval = fread(zline, sizeof(char), s_z, rifffd);
					if(retval != s_z)
					{
						fprintf(stderr, "%s Data read error <%s> %d != %d\n", __FUNCTION__, riffFname, retval,  s_z);
						goto processingFault;
					}
					// insert line into space
					retval = FD_zlineInsert(zline, i+off_x, j+off_y, off_z, s_z);
					if(retval)
					{
						fprintf(stderr, "%s Data write error into GPU space at (%d,%d,%d) size:%d returned %d\n", __FUNCTION__,  i+off_x, j+off_y, off_z, s_z, retval);
						goto processingFault;
					}
				}
			}
			free(zline);
		//	break;
		}
		else
		{ // not my type of chunk
			fprintf(stderr, "Unrecognized chunk\n");
			// if not a chunk, we are lost, break out of processing
			if(!isalnum(chnkName[0]) || !isalnum(chnkName[1]) || !isalnum(chnkName[2]) || !isalnum(chnkName[3]) )
				break;
			fseek(rifffd, chnkSize, SEEK_CUR);
			continue;
		}
	}
processingFault:
	fclose(rifffd);
	return(0);
}

// main processing of xml files here
extern int FP_ProcessFile(char* fname, int verbose, int silent)
{
	error_t retval = 0;
	xmlDocPtr boardDoc;
	xmlDocPtr xemDoc;
	char* xemFilename;

	// Load XML document
	boardDoc = xmlParseFile(fname);
	if (boardDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse File \"%s\"\n", fname);
		return(-1);
	}

	// determine xml file typ by looking at root level doc	
//	xmlNodeSetPtr boardDocFrag = XPU_GetNodeSet(boardDoc, XPATH_XEM_BOARD_DOC);
//	if(boardDocFrag == NULL)
//		goto processingFault;
	
	// create the materials table
	xmlNodeSetPtr xnsMaterials  = XPU_GetNodeSet(boardDoc, XPATH_XEM_MATERIALS);
	if(xnsMaterials != NULL)
		retval = MATRL_CreateTableFromNodeSet(xnsMaterials);
		
	if(retval)
		goto processingFault;

	MATRL_DumpAll();

       // get xem filename
	xemFilename = getXemFilename(boardDoc, fname);
	if(xemFilename != NULL)
	{
		fprintf(stdout, "%s\n",xemFilename);

		// parse xem file
		xemDoc = xmlParseFile(xemFilename);
		if(xemDoc == NULL)
		{
			fprintf(stderr, "Error: unable to parse file \"%s\"\n", xemFilename);
			return(-1);
		}

		// get width, in voxels (pixels)
		xmlChar* cWidth = XPU_SimpleLookup(xemDoc,XPATH_NELMA_WIDTH);
		if(cWidth)
		{
			gint width = strtol((char*)cWidth,NULL,10);
			xmlFree(cWidth);
			fprintf(stdout,"w:%d\n",width);
		}

		// get height, in voxels (pixels)
		xmlChar* cHeight = XPU_SimpleLookup(xemDoc,XPATH_NELMA_HEIGHT);
		if(cHeight)
		{
			gint height = strtol((char*)cHeight,NULL,10);
			xmlFree(cHeight);
			fprintf(stdout,"h:%d\n", height);
		}


		// get the pixel resolution, in meters per pixel
		double res = XPU_GetDouble(xemDoc, XPATH_NELMA_RES);
		if(!isnan(res))
		{
			xmlChar* units = XPU_SimpleLookup(xemDoc, XPATH_NELMA_RES_UNITS);
			res = MATRL_ScaleToMeters(res, (char*)units);
			fprintf(stdout, "adjusted res: %g (meters/pixel)\n", res);
		}
	}

	// get the layers from the board document
	xmlNodeSetPtr xnsLayers  = XPU_GetNodeSet(boardDoc, XPATH_XEM_LAYERS);
	if(xnsLayers == NULL)
		goto processingFault;
		
	char* riffFilename = getRiffFilename(boardDoc, fname);
	if(riffFilename != NULL)
	{
		FP_ReadRiff(riffFilename);
		
	}


processingFault:
	return(retval);
}

extern void FP_MakeDrill(int xCenter, int yCenter, int outerRadius, int innerRadius, int zStart, int zLen, char matIndex)
{
	int rowSize = outerRadius*2+1;
	int colSize = rowSize;	// circles are round, but ExtrudeZ doesn't care
	int size = rowSize*colSize*sizeof(char);
	char* pTemplate = (char*)malloc(size);
	memset(pTemplate,0,size);
	int i;
	int j;
	char airIndex = 3;
	for(i=0;i<=rowSize/2;i++)
	{
		for(j=0;j<=colSize/2;j++)
		{
			int r;
			r = sqrt(i*i+j*j)+0.5;
			if(r>=innerRadius && r<=outerRadius)
			{
			
				int x = i;
				int y = j;
				int xOff = outerRadius;
				int yOff = outerRadius;
				// compute for one quadrant, apply to four quadrants
				int index = (x+xOff) + (y+yOff)*rowSize;
				pTemplate[index] = matIndex;
				index = (x+xOff) + (-y+yOff)*rowSize;
				pTemplate[index] = matIndex;
				index = (-x+xOff) + (y+yOff)*rowSize;
				pTemplate[index] = matIndex;
				index = (-x+xOff) + (-y+yOff)*rowSize;
				pTemplate[index] = matIndex;
			}
			else if(r<innerRadius)
			{
			
				int x = i;
				int y = j;
				int xOff = outerRadius;
				int yOff = outerRadius;
				// compute for one quadrant, apply to four quadrants
				int index = (x+xOff) + (y+yOff)*rowSize;
				pTemplate[index] = airIndex;
				index = (x+xOff) + (-y+yOff)*rowSize;
				pTemplate[index] = airIndex;
				index = (-x+xOff) + (y+yOff)*rowSize;
				pTemplate[index] = airIndex;
				index = (-x+xOff) + (-y+yOff)*rowSize;
				pTemplate[index] = airIndex;
			}
		}			
	}
#if 0
	printf("rowSize: %d\n",rowSize);
	for(i=0;i<size;i++)
	{
		if(pTemplate[i])
			printf("X");
		else
			printf(".");
		if(i%rowSize==rowSize-1)
			printf("\n");
	}
#endif
//	int xDim, int yDim, int xCenter, int yCenter, int zStart, int zEnd
	SimulationSpace_ExtrudeZ(pTemplate, rowSize, colSize, xCenter, yCenter, zStart, zLen );
	free(pTemplate);	
}

extern void FP_MakeRectangleX(int yCenter, int zCenter, int yLen, int zLen, int xStart, int xLen, char matIndex)
{
	int rowSize = yLen;
	int colSize = zLen;
	int size = rowSize*colSize*sizeof(char);
	char* pTemplate = (char*)malloc(size);
	memset(pTemplate,matIndex,size);
//	int xDim, int yDim, int xCenter, int yCenter, int zStart, int zEnd
	SimulationSpace_ExtrudeX(pTemplate, rowSize, colSize, yCenter, zCenter, xStart, xLen );
	free(pTemplate);	
}


extern void FP_MakeRectangleY(int xCenter, int zCenter, int xLen, int zLen, int yStart, int yLen, char matIndex)
{
	int rowSize = xLen;
	int colSize = zLen;
	int size = rowSize*colSize*sizeof(char);
	char* pTemplate = (char*)malloc(size);
	memset(pTemplate,matIndex,size);
//	int xDim, int yDim, int xCenter, int yCenter, int zStart, int zEnd
	SimulationSpace_ExtrudeY(pTemplate, rowSize, colSize, xCenter, zCenter, yStart, yLen );
	free(pTemplate);	
}


extern void FP_MakeRectangleZ(int xCenter, int yCenter, int xLen, int yLen, int zStart, int zLen, char matIndex)
{
	int rowSize = xLen;
	int colSize = yLen;
	int size = rowSize*colSize*sizeof(char);
	char* pTemplate = (char*)malloc(size);
	memset(pTemplate,matIndex,size);
//	int xDim, int yDim, int xCenter, int yCenter, int zStart, int zEnd
	SimulationSpace_ExtrudeZ(pTemplate, rowSize, colSize, xCenter, yCenter, zStart, zLen );
	free(pTemplate);	
}


int FP_ProcessDrillNodeSet(xmlNodeSetPtr xnsPtr, int z1, int z2, int plateThickness)
{
	int i;
	int isPlated = 1;
	if( (xnsPtr == NULL) || (z1==z2))
	{
		fprintf(stderr, "%s General Input fault\n", __FUNCTION__);
		return(-1);
	}

	for(i = 0; i<xnsPtr->nodeNr; i++)
	{
		//              char xpathString[0x400];
		xmlNodePtr currDrill = xnsPtr->nodeTab[i];
		if(currDrill == NULL)
		{
			return(-2);
		}
/*
<drill id="D0">
<dia_inches>0.012</dia_inches>
<radius>6</radius>
<pos type="plated">662,148</pos>
</drill>
*/		        											// get the drill name
		xmlChar* drillRadius = XPU_LookupFromNode(currDrill, "./radius/text()");
		fprintf(stdout, "%s:%d radius:<%s>  \n", __FUNCTION__, i, drillRadius);
		int radius = strtol((char*)drillRadius,NULL,10);

		xmlNodeSetPtr xnsPos = XPU_GetNodeSetFromNode(currDrill, "./pos");
		if(xnsPos == NULL)
		{
			fprintf(stdout, "%s: NULL pos set\n", __FUNCTION__);
			return(-3);
		}
		 
		int j;
		for(j=0;j<xnsPos->nodeNr;j++)
		{
			xmlNodePtr currPos = xnsPos->nodeTab[j];
			if(currPos == NULL)
			{
				return(-2);
			}
			xmlChar* xpos = XPU_LookupFromNode(currPos, "./text()");
//			fprintf(stdout, "%s ",(char*)xpos);
			xmlChar* xtype = XPU_LookupFromNode(currPos, "@type");
//			fprintf(stdout, "%s \n",(char*)xtype);

			char* cval = strtok((char*)xpos, ",");
			int xp = strtol((char*)cval,NULL,10);
			cval = strtok(NULL, ",");
			int yp = strtol((char*)cval,NULL,10);
			
			if(strstr((char*)xtype,"unplated")!=NULL)
			{
				isPlated = 0;
			}
			else if(strstr((char*)xtype,"plated")!=NULL)
			{
				isPlated = 1;
			}
			
			fprintf(stdout,"%s x:%d, y:%d, z1:%d, z2:%d, r:%d, t:%d, %s\n", __FUNCTION__, xp,yp,z1,z2,radius,plateThickness, (char*)xtype);
			
			if(isPlated == 1)
			{ // a plated hole consists of two cylinders, one  copper the other is air
					// 7 the geometery type (cyl)
					//      x1 y1 z1 x2 y2 z2 rad 
//				fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius, COPPER_CYL_INFO);
//				if((radius-plateThickness) > 0)
//					fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius-plateThickness, AIR_CYL_INFO);
				FP_MakeDrill(xp, yp, radius, radius-plateThickness, z1, z2-z1, 2);
			}
			else
			{ // an unplaed hole consists of a cylinder of air
//				fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius, AIR_CYL_INFO);
				FP_MakeDrill(xp, yp, radius, radius, z1, z2-z1, 3);
			}
			
			        			
		}
	}
	return(0);
}

