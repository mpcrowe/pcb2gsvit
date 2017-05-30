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

/*----------------------------------------------------------------------------
*        Internal definitions
*----------------------------------------------------------------------------*/
#define MAX_FILENAME 0x200
#define SVN_REV "found on github at https://github.com/mpcrowe/pcb2gsvit.git"

#define XPATH_XEM_NAME "/boardInformation/nelmaExport/text()"
#define XPATH_XEM_OUTPUT_FILENAME "/boardInformation/gsvit/mediumLinearFilename/text()"
#define XPATH_XEM_MATERIALS "/boardInformation/materials/material"
#define XPATH_XEM_LAYERS "/boardInformation/boardStackup/layer"
#define XPATH_XEM_OUTLINE "/boardInformation/boardStackup/layer[name/text()='outline']/material/text()"

#define XPATH_NELMA_WIDTH "/nelma/space/width/text()"
#define XPATH_NELMA_HEIGHT "/nelma/space/height/text()"
#define XPATH_NELMA_RES	"/nelma/space/resolution/text()"
#define XPATH_NELMA_RES_UNITS	"/nelma/space/resolution/@units"


/*----------------------------------------------------------------------------
*        Local variables
*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/
char* getNelmaFilename(xmlDocPtr doc, const char* parentDocName);
char* getFilenamePath( const char* parentDocName);
char* getFilename(xmlDocPtr doc, const char* parentDocName, char* dest, const char* xpath);
char* getNelmaFilename(xmlDocPtr doc, const char* parentDocName);
char* getMediumLinearOutputFilename(xmlDocPtr doc, const char* parentDocName);
 	 	   	 	  
int execute_conversion(const char* filename);
static void usage(const char *name);


/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/

char* getFilenamePath( const char* parentDocName)
{
	static char cwd[0x400];

	char* preEnd;

	if( getcwd(cwd, sizeof(cwd)) == NULL)
	{
		perror("getcwd() error");
		return(NULL);
	}

	preEnd = strrchr(parentDocName, '/');
	if(preEnd != NULL)
	{
		char* pstr = (char*)parentDocName;
		int n = strlen(cwd);
		char* pdest = &cwd[n];
		*pdest++ = '/';
		while(pstr != preEnd)
		{
			*pdest++ = *pstr++;
		}
		*pdest = '\0';
	}

	return(cwd);
}


char* getFilename(xmlDocPtr doc, const char* parentDocName, char* dest, const char* xpath)
{
	xmlChar* keyword = XPU_SimpleLookup(doc, (char*)xpath);
	if( keyword == NULL)
		return( NULL);
	char* cwd = getFilenamePath(parentDocName);
	if(cwd == NULL)
		return(NULL);
	sprintf(dest, "%s/%s",cwd, keyword );
	xmlFree(keyword);
	return(dest);
}

char* getLayerFilename(const char* nelmaName, char* dest, char* basename);
char* getLayerFilename(const char* nelmaName, char* dest, char* basename)
{
	strcpy(dest,nelmaName);
	char* end = strcasestr(dest, ".xem");
	sprintf(end, ".%s.png", basename );
	return(dest);
}


char* getNelmaFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return(getFilename(doc, parentDocName, fullName, XPATH_XEM_NAME) );
}


char* getMediumLinearOutputFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return( getFilename(doc, parentDocName, fullName, XPATH_XEM_OUTPUT_FILENAME));
}


int execute_conversion(const char* filename)
{
	xmlDocPtr boardDoc;
	xmlDocPtr nelmaDoc;
	char* nelmaFilename;
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

	// get nelma filename
	nelmaFilename = getNelmaFilename(boardDoc, filename);
	if(nelmaFilename == NULL)
	{
		goto processingFault;
	}
	fprintf(stdout, "%s\n",nelmaFilename);

	// parse nelma file
	nelmaDoc = xmlParseFile(nelmaFilename);
	if(nelmaDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", nelmaFilename);
		return(-1);
	}

	// get width, in voxels (pixels)
	xmlChar* cWidth = XPU_SimpleLookup(nelmaDoc,XPATH_NELMA_WIDTH);
	if(cWidth == NULL)
		goto processingFault;
	gint width = strtol((char*)cWidth,NULL,10);
	xmlFree(cWidth);

	// get height, in voxels (pixels)
	xmlChar* cHeight = XPU_SimpleLookup(nelmaDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
	gint height = strtol((char*)cHeight,NULL,10);
	xmlFree(cHeight);

	fprintf(stdout,"w:%d: h:%d:\n",width, height);

	// get the pixel resolution, in meters per pixel
	double res = XPU_GetDouble(nelmaDoc, XPATH_NELMA_RES);
	if(isnan(res))
		goto processingFault;
	xmlChar* units = XPU_SimpleLookup(nelmaDoc, XPATH_NELMA_RES_UNITS);
	res = MATRL_ScaleToMeters(res, (char*)units);
	fprintf(stdout, "adjusted res: %g\n", res);

	// create the materials table
	xmlNodeSetPtr xnsMaterials  = XPU_GetNodeSet(boardDoc, XPATH_XEM_MATERIALS);
	if(xnsMaterials == NULL)
		goto processingFault;

	if(MATRL_CreateTableFromNodeSet(xnsMaterials) !=0)
		goto processingFault;
	
	MATRL_DumpAll();
	
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
	getLayerFilename(nelmaFilename, layerFname, "outline");
	fprintf(stdout, "outline fname: %s\n",layerFname);
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
	
	fprintf(stdout, "layer processing\n");	
	// for each layer, create a unique layer unless it already exists (fill and outline layers)
	GList* gLayers = NULL;
	for(i = 0; i<xnsLayers->nodeNr; i++)
	{
		fRect* fRectCurrent;
//		char xpathString[0x400];
		xmlNodePtr currLayer = xnsLayers->nodeTab[i];
		if(currLayer == NULL)
		{
			xmlFreeDoc(nelmaDoc);
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
			getLayerFilename(nelmaFilename, layerFname, (char*)layerName);
			fprintf(stdout, ": %s\n",layerFname);
			if(LAYER_ReadPng(layerFname))
			{
				fprintf(stdout, "warning, no layer found, using default fill(air) for board values\n");
			}
			else
			{
				LAYER_ProcessLayer(fRectCurrent, mIndex);
			}	

			
// process a copper layer here			
			
		}
		for(k=0; k< zVoxelCount; k++)
			gLayers = g_list_prepend(gLayers, fRectCurrent);
			
		
//		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/relativePermittivity/text()", materialName);
//		xmlChar* cEr = XPU_SimpleLookup(boardDoc, xpathString);
//		if(cEr == NULL)
//		{
//			goto processingFault;
//		}
//		fprintf(stdout, "Er: %s\t",  cEr);
//
//		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/conductivity/text()", materialName);
//		xmlChar* cCond = XPU_SimpleLookup(boardDoc, xpathString);
//		if(cCond == NULL)
//		{
//			goto processingFault;
//		}
//
//		fprintf(stdout, "Conductivity: %s\n",  cCond);

		fprintf(stdout, "\n");
//		createLayer(width, height, )
	}

	gLayers = g_list_reverse(gLayers);
	gint depth =  g_list_length(gLayers);

	float* fSlice = (float*)malloc(sizeof(float)*depth);
	if( fSlice == NULL)
		goto processingFault;
	
	
	fprintf(stdout, "\nLayer processing complete\nOpening output\n");
	// open the Medium Linear Output file for gsvit
	char* mlFname = getMediumLinearOutputFilename(boardDoc, filename);
	if(mlFname == NULL)
		goto processingFault;
	fprintf(stdout,"medium linear filename: %s\n", mlFname);
	FILE* mlfd = fopen(mlFname, "w");
	if(mlfd == NULL)
	{
		fprintf(stderr, "Unable to open <%s>\n", mlFname);
		goto processingFault;
	}
	fprintf(stdout, "x:%d, y:%d z:%d  0x%x 0x%x 0x%x\n", width, height, depth, width, height, depth);
	fwrite(&width, sizeof(gint), 1, mlfd);
	fwrite(&height, sizeof(gint), 1, mlfd);
	retval = fwrite(&depth, sizeof(gint) ,1, mlfd);
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



	fclose(mlfd);
	fprintf(stderr, "processing complete, no errors encountered\n");
	xmlFreeDoc(nelmaDoc);
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
	if(execute_conversion(argv[1] ) < 0)
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

