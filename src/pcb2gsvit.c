/*----------------------------------------------------------------------------
*        Headers
*----------------------------------------------------------------------------*/
#define _GNU_SOURCE
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <unistd.h>
#include <glib.h>  // for  using GList

#include "frect.h"
#include "xpu.h"
#include "layer.h"
#include "med_vect.h"
#include "file_edit.h"

/*----------------------------------------------------------------------------
*        Internal definitions
*----------------------------------------------------------------------------*/
#define MAX_FILENAME 0x200
#define SVN_REV "found on github at https://github.com/mpcrowe/pcb2gsvit.git"
#include "xpathConsts.h"


/*----------------------------------------------------------------------------
*        Local variables
*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/
char* getXemFilename(xmlDocPtr doc, const char* parentDocName);
char* getMediumLinearOutputFilename(xmlDocPtr doc, const char* parentDocName);
 	 	   	 	  
int execute_conversion(const char* filename, int verbose);
static void usage(const char *name);


/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/

void generateParFileEditRules(void)
{
	FE_AppendRule("POOL", "llelkd", NULL);
	FE_AppendRule("MEDIUM_LINEAR", "medLin.bin", NULL);
	FE_AppendRule("MEDIUM_VECTOR", "matvect.txt", NULL);
}



char* getLayerFilename(const char* xmlName, char* dest, char* basename);
char* getLayerFilename(const char* xmlName, char* dest, char* basename)
{
	strcpy(dest,xmlName);
	char* end = strcasestr(dest, ".xem");
	sprintf(end, ".%s.png", basename );
	return(dest);
}


char* getXemFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return(XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_NAME) );
}


char* getMediumLinearOutputFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return( XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_OUTPUT_FILENAME));
}

char* getRiffOutputFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return( XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_RIFF_FILENAME));
}


int execute_conversion(const char* filename, int verbose)
{
	xmlDocPtr boardDoc;
	xmlDocPtr xemDoc;
	char* xemFilename;
	int i;
	int j;
	int k;
	int retval;
	// Load XML document
	boardDoc = xmlParseFile(filename);
	if (boardDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", filename);
		return(-1);
	}

	// get xem filename
	xemFilename = getXemFilename(boardDoc, filename);
	if(xemFilename == NULL)
	{
		goto processingFault;
	}
	if(verbose)
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
	if(cWidth == NULL)
		goto processingFault;
	int32_t width = strtol((char*)cWidth,NULL,10);
	xmlFree(cWidth);

	// get height, in voxels (pixels)
	xmlChar* cHeight = XPU_SimpleLookup(xemDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
	int32_t height = strtol((char*)cHeight,NULL,10);
	xmlFree(cHeight);

	if(verbose)
		fprintf(stdout,"w:%d: h:%d:\n",width, height);

	// get the pixel resolution, in meters per pixel
	double res = XPU_GetDouble(xemDoc, XPATH_NELMA_RES);
	if(isnan(res))
		goto processingFault;
	xmlChar* units = XPU_SimpleLookup(xemDoc, XPATH_NELMA_RES_UNITS);
	res = MATRL_ScaleToMeters(res, (char*)units);
	if(verbose)
		fprintf(stdout, "adjusted res: %g\n", res);

	// create the materials table
	retval = MATRL_CreateTable( boardDoc, verbose);
	if(retval)
		goto processingFault;

	
	xmlChar* outlineMaterial = XPU_SimpleLookup(boardDoc, XPATH_XEM_OUTLINE);
	if(outlineMaterial == NULL)
	{
//		fprintf(stderr, "path <%s>\n",XPATH_XEM_OUTLINE);
		goto processingFault;
	}
	
	// create the layers that are used as a template
	// when there is nothing there (air, fill, default)
	fRect* fillLayer = FRECT_New(width, height);
	FRECT_Fill(fillLayer, 0);
	
	// the board outline layer is also special
	// if it's not there, we assume that the
	// board dimensions are the same as the height and width
	fRect* boardOutlineLayer = NULL;
	char layerFname[0x400];
	getLayerFilename(xemFilename, layerFname, "outline");
//	fprintf(stdout, "outline fname: %s\n",layerFname);
	if(LAYER_ReadPng(layerFname))
	{
		fprintf(stdout, "warning, no outline layer found, using default fill(air) for board values\n");
		boardOutlineLayer = FRECT_Clone(fillLayer);
	}
	else
	{
		boardOutlineLayer = FRECT_Clone(fillLayer);
		LAYER_ProcessOutline(boardOutlineLayer, MATRL_GetIndex((char*)outlineMaterial));
	}	

	
	// get the layers from the board document
	xmlNodeSetPtr xnsLayers  = XPU_GetNodeSet(boardDoc, XPATH_XEM_LAYERS);
	if(xnsLayers == NULL)
		goto processingFault;
	
//	fprintf(stdout, "layer processing\n");	
	// for each layer, create a unique layer unless it already exists (fill and outline layers)
	GList* gLayers = NULL;
	int zVoxelTotal = 0;
	int zVoxelBottom  = 0;
	int zVoxelTop = 0;
	for(i = 0; i<xnsLayers->nodeNr; i++)
	{
		fRect* fRectCurrent;
//		char xpathString[0x400];
		xmlNodePtr currLayer = xnsLayers->nodeTab[i];
		if(currLayer == NULL)
		{
			xmlFreeDoc(xemDoc);
			xmlFreeDoc(boardDoc);
			return(-1);
		}

		// get the layer name
		xmlChar* layerName = XPU_LookupFromNode(currLayer, "./name/text()");
		fprintf(stdout, "%d name:<%s>  \t", i, layerName);

		// get the base type fill or outline or null
		xmlChar* baseType = XPU_LookupFromNode(currLayer, "./baseType/text()");
		fprintf(stdout, "%d name:<%s>  \t", i, layerName);
		
		// get the material name and material index (from the name)
		xmlChar* materialName = XPU_LookupFromNode(currLayer, "./material/text()");
		if(materialName == NULL)
		{
			fprintf(stderr, "\nError, no material name specified\n");
			goto processingFault;
		}
		fprintf(stdout, "material:<%s> \t",  materialName);
		int mIndex = MATRL_GetIndex((char*)materialName);
		if(mIndex < 0)
		{
			goto processingFault;
		}
		fprintf(stdout, "\tmaterial Index: %d\n", mIndex);

		// get the thickness of the layer
		xmlChar* cThickness = XPU_LookupFromNode(currLayer, "./thickness/text()");
		if(cThickness == NULL)
		{
			cThickness = MATRL_DefaultThickness(mIndex);
			fprintf(stdout, "WARN, using default thickness %s\n",cThickness);
		}
		if(cThickness == NULL)
		{
			fprintf(stderr, "ERROR, no thickness or default thickness defined\n");
			goto processingFault;
		}
		int zVoxelCount = MATRL_StringToCounts((char*)cThickness, res);
		if(zVoxelCount <=0)
		{
			goto processingFault;
		}
		fprintf(stdout, "\tvoxel count, z-axis: %d\n", zVoxelCount);
		
		if(baseType ==NULL)
		{ // no base line type, must be a premade type
			if(strstr("outline", (char*)layerName) != NULL)
				fRectCurrent = boardOutlineLayer;
			else
				fRectCurrent = fillLayer;
		}
		else
		{
			if(strstr("outline", (char*)baseType) != NULL)
				fRectCurrent = FRECT_Clone(boardOutlineLayer);
			else
				fRectCurrent = FRECT_Clone(fillLayer);
			fprintf(stdout, "WARNING: need to process a real layer of traces for: %s\n", layerName);
			getLayerFilename(xemFilename, layerFname, (char*)layerName);
			fprintf(stdout, ": %s\n",layerFname);
			if(LAYER_ReadPng(layerFname))
			{
				fprintf(stdout, "warning, no layer found, using default fill(air) for board values\n");
			}
			else
			{
				LAYER_ProcessLayer(fRectCurrent, mIndex);
			}	
		}
		for(k=0; k< zVoxelCount; k++)
			gLayers = g_list_prepend(gLayers, fRectCurrent);
printf("bottom\n");

		if(strstr("bottom",  (char*)layerName) != NULL)
		{
			zVoxelBottom = zVoxelTotal;
		}
		zVoxelTotal += zVoxelCount;
		if(strstr("top",  (char*)layerName) != NULL)
		{
			zVoxelTop = zVoxelTotal;
		}
		fprintf(stdout, "\n");
	}

	gLayers = g_list_reverse(gLayers);
	int32_t depth =  g_list_length(gLayers);

	float* fSlice = (float*)malloc(sizeof(float)*depth);
	if( fSlice == NULL)
		goto processingFault;

	char* cSlice = (char*)malloc(sizeof(char)*depth);
	if( cSlice == NULL)
		goto processingFault;
	
	
	fprintf(stdout, "\nLayer processing complete\nOpening output\n");
	// open the Medium Linear Output file for gsvit
	char* mlFname = getMediumLinearOutputFilename(boardDoc, filename);
	if(mlFname != NULL)
	{
#define OUTPUT_MED_LIN 1
#ifdef OUTPUT_MED_LIN
		fprintf(stdout,"medium linear filename: %s\n", mlFname);
		FILE* mlfd = fopen(mlFname, "w");
		if(mlfd == NULL)
		{
			fprintf(stderr, "Unable to open <%s>\n", mlFname);
			goto processingFault;
		}
		fprintf(stdout, "x:%d, y:%d z:%d  0x%x 0x%x 0x%x\n", width, height, depth, width, height, depth);
		fwrite(&width, sizeof(int32_t), 1, mlfd);
		fwrite(&height, sizeof(int32_t), 1, mlfd);
		retval = fwrite(&depth, sizeof(int32_t) ,1, mlfd);
		if(retval != 1)
		{
			fprintf(stderr, "Write error %d!= 1\n",retval);
			goto processingFault;
		}
		fprintf(stdout, "starting Er\n");
		fprintf(stdout, "size x:%d, y:%d z:%d\n",width, height, depth);
		for(i=0; i<width; i++)
		{
			for(j=0; j<height; j++)
			{
				float* pSlice = fSlice;
				GList *l;
				for (l = gLayers; l != NULL; l = l->next)
				{
					// do something with l->data
					int index = ((fRect*)(l->data))->data[i][j];
					*pSlice++ = MATRL_Er(index);
				}
				retval = fwrite(fSlice, sizeof(float), depth, mlfd);
				if(retval != depth)
				{
					fprintf(stderr, "file write error %d!=%d", retval, depth);
					goto processingFault;
				}
			}
		}

		fprintf(stdout, "starting conductivity\n");		
		for(i=0; i<width; i++)
		{
			for(j=0; j<height; j++)
			{
				float* pSlice = fSlice;
				GList *l;
				for (l = gLayers; l != NULL; l = l->next)
				{
					// do something with l->data
					int index = ((fRect*)(l->data))->data[i][j];
					*pSlice++ = MATRL_Cond(index);
				}
				retval = fwrite(fSlice, sizeof(float), depth, mlfd);
				if(retval != depth)
				{
					fprintf(stderr, "file write error %d!=%d", retval, depth);
					goto processingFault;
				}
				
			}
		}
		fprintf(stdout, "starting Ur\n");
		fprintf(stdout, "size x:%d, y:%d z:%d\n",width, height, depth);
		for(i=0; i<width; i++)
		{
			for(j=0; j<height; j++)
			{
				float* pSlice = fSlice;
				GList *l;
				for (l = gLayers; l != NULL; l = l->next)
				{
					// do something with l->data
					int index = ((fRect*)(l->data))->data[i][j];
					*pSlice++ = 1.0;
					//				*pSlice++ = MATRL_Ur(index);
				}
				retval = fwrite(fSlice, sizeof(float), depth, mlfd);
				if(retval != depth)
				{
					fprintf(stderr, "file write error %d!=%d", retval, depth);
					goto processingFault;
				}
			}
		}
		fprintf(stdout, "starting magnetic conductivity (susceptibility)\n");
		fprintf(stdout, "size x:%d, y:%d z:%d\n",width, height, depth);
		for(i=0; i<width; i++)
		{
			for(j=0; j<height; j++)
			{
				float* pSlice = fSlice;
				GList *l;
				for (l = gLayers; l != NULL; l = l->next)
				{
					int index = ((fRect*)(l->data))->data[i][j];
					//				*pSlice++ = MATRL_Sus(index);
					*pSlice++ = 0.0;
				}
				retval = fwrite(fSlice, sizeof(float), depth, mlfd);
				if(retval != depth)
				{
					fprintf(stderr, "file write error %d!=%d", retval, depth);
					goto processingFault;
				}
			}
		}

		fclose(mlfd);
#endif	
	}
	
	// open the riff Output file for efdtd
	char* riffFname = getRiffOutputFilename(boardDoc, filename);
	if(riffFname != NULL)
	{
		fprintf(stdout,"riff filename: %s\n", riffFname);
		FILE* rifffd = fopen(riffFname, "w");
		if(rifffd == NULL)
		{
			fprintf(stderr, "Unable to open <%s>\n", riffFname);
			goto processingFault;
		}

		int32_t chnkSize = width*height*depth*sizeof(char) + sizeof(int32_t)*3*2;
		int32_t offset_x = 0;
		int32_t offset_y = 0;
		int32_t offset_z = 0;

		fprintf(stdout, "x:%d, y:%d z:%d  0x%x 0x%x 0x%x chnk:%d\n", width, height, depth, width, height, depth, chnkSize);

		// write out header, ID, chnk size, sz_x sz_y, sz_z, off_x, off_y, off_z, data....
		retval = fprintf(rifffd,"fdtd");
		retval += fwrite(&chnkSize, sizeof(int32_t), 1, rifffd);
		retval += fwrite(&width, sizeof(int32_t), 1, rifffd);
		retval += fwrite(&height, sizeof(int32_t), 1, rifffd);
		retval += fwrite(&depth, sizeof(int32_t) ,1, rifffd);
		retval += fwrite(&offset_x, sizeof(int32_t) ,1, rifffd);
		retval += fwrite(&offset_y, sizeof(int32_t) ,1, rifffd);
		retval += fwrite(&offset_z, sizeof(int32_t) ,1, rifffd);
		if(retval != 11)
		{
			fprintf(stderr, "Header write error <%s> %d != %ld\n", riffFname, retval,  sizeof(int32_t)*8);
			goto processingFault;
		}
		

		for(i=0; i<width; i++)
		{
			for(j=0; j<height; j++)
			{
				char* pSlice = cSlice;
				GList *l;
				for (l = gLayers; l != NULL; l = l->next)
				{
					// do something with l->data
					int index = ((fRect*)(l->data))->data[i][j];
					*pSlice++ = (char)index;
				}
				retval = fwrite(cSlice, sizeof(char), depth, rifffd);
				if(retval != depth)
				{
					fprintf(stderr, "file write error %d!=%d", retval, depth);
					goto processingFault;
				}
			}
		}
		fclose(rifffd);
		
	}
	
	
	xmlNodeSetPtr xnsDrills  = XPU_GetNodeSet(xemDoc, XPATH_NELMA_DRILLS);
	

	retval = MV_ProcessDrillNodeSet(stdout, xnsDrills, zVoxelBottom, zVoxelTop, 2);
	
	fprintf(stderr, "processing complete, no errors encountered\n");
	xmlFreeDoc(xemDoc);
	xmlFreeDoc(boardDoc);
	return(0);

processingFault:
	fprintf(stderr, "processing fault\n");
	xmlFreeDoc(boardDoc);
	return(-1);
}


static void usage(const char *name)
{
	fprintf(stderr, "Usage: %s <xml-file>\n", name);
	fprintf(stderr, "where <xml-file> is a valid description of the pcb to be\n");
	fprintf(stderr, "analyzed, see https://github.com/mpcrowe/pcb2gsvit.git for how to use\n");
}


/*----------------------------------------------------------------------------
*        Exported functions
*----------------------------------------------------------------------------*/

// a command line shell interface
// all the work is done in a another function
int main(int argc, char **argv)
{
	int verbose = 1;
	// Parse command line and process file
	if((argc < 1) || (argc > 2))
	{
		fprintf(stderr, "Error: wrong number of arguments.\n");
		usage(argv[0]);
		return(-1);
	}

	// Init libxml
	xmlInitParser();
	LIBXML_TEST_VERSION

	// Do the main job
	if(execute_conversion(argv[1], verbose ) < 0)
	{
		usage(argv[0]);
		return(-1);
	}

	// Shutdown libxml
	xmlCleanupParser();

	// this is to debug memory for regression tests
//	xmlMemoryDump();
	return(0);
}

