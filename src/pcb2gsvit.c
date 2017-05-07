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
double scaleToMeters(double val, char* units);
 	 	   	 	  
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


double scaleToMeters(double val, char* units)
{
	if(units == NULL)
		return(val);
	if(strncmp(units, "mm", 2) == 0)
		return(val/1000);
	if(strncmp(units, "mil", 3) == 0)
		return(val*2.54e-5);
	if(strncmp(units, "in", 2) == 0)
		return(val*0.0254);
	if(strncmp(units, "m", 1) == 0)
		return(val);
	if(strncmp(units, "ft", 2) == 0)
		return(val*0.3048);
	fprintf(stderr, "Unknown unit prefix of <%s>\n",units);
	return(val);
}


int execute_conversion(const char* filename)
{
	xmlDocPtr boardDoc;
	xmlDocPtr nelmaDoc;
	char* nelmaFilename;
	int i;

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
	uint32_t width = strtol((char*)cWidth,NULL,10);
	xmlFree(cWidth);

	// get height, in voxels (pixels)
	xmlChar* cHeight = XPU_SimpleLookup(nelmaDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
	uint32_t height = strtol((char*)cHeight,NULL,10);
	xmlFree(cHeight);

	fprintf(stdout,"w:%d: h:%d:\n",width, height);

	// get the pixel resolution, in meters per pixel
	double res = XPU_GetDouble(nelmaDoc, XPATH_NELMA_RES);
	if(isnan(res))
		goto processingFault;
	xmlChar* units = XPU_SimpleLookup(nelmaDoc, XPATH_NELMA_RES_UNITS);
	res = scaleToMeters(res, (char*)units);
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
		fprintf(stdout, "warning, no outline layer found\n");
	}
	else
	{
		boardOutlineLayer = FRECT_Clone(fillLayer);
		LAYER_ProcessOutline(boardOutlineLayer, MATRL_GetIndex((char*)outlineMaterial));
	}	
	
	


	xmlNodeSetPtr xnsLayers  = XPU_GetNodeSet(boardDoc, XPATH_XEM_LAYERS);
	if(xnsLayers == NULL)
		goto processingFault;


	fprintf(stdout, "opening output\n");

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

	for(i = 0; i<xnsLayers->nodeNr; i++)
	{
		char xpathString[0x400];
		xmlNodePtr currLayer = xnsLayers->nodeTab[i];
		if(currLayer == NULL)
		{
			fclose(mlfd);
			xmlFreeDoc(nelmaDoc);
			xmlFreeDoc(boardDoc);
			return(-1);
		}

		xmlChar* layerName = XPU_LookupFromNode(currLayer, "./name/text()");
		fprintf(stdout, "%d name:<%s>  \t", i, layerName);
		xmlChar* materialName = XPU_LookupFromNode(currLayer, "./material/text()");
		if(materialName == NULL)
		{
			fprintf(stderr, "\nError, no material name specified\n");
			fclose(mlfd);
			goto processingFault;
		}
		fprintf(stdout, "material:<%s> \t",  materialName);



		xmlChar* cThickness = XPU_LookupFromNode(currLayer, "./thickness/text()");
		if(cThickness != NULL)
		{
			fprintf(stdout, "thickness:<%s>", cThickness);
		}
		fprintf(stdout, "\n");

		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/relativePermittivity/text()", materialName);
		xmlChar* cEr = XPU_SimpleLookup(boardDoc, xpathString);
		if(cEr == NULL)
		{
			fclose(mlfd);
			goto processingFault;
		}
		fprintf(stdout, "Er: %s\n",  cEr);

		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/conductivity/text()", materialName);
		xmlChar* cCond = XPU_SimpleLookup(boardDoc, xpathString);
		if(cCond == NULL)
		{
			fclose(mlfd);
			goto processingFault;
		}

		fprintf(stdout, "Conductivity: %s\n",  cCond);
//		createLayer(width, height, )


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

