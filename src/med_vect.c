// med_vect
// processes an xml drill fragment to produce a medium vector file format
// of cylinders


#include <stdio.h>
#include <string.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include "med_vect.h"
#include "xpu.h"

// allocates memory for a new frect instance
FILE* MV_Open(char* name)
{
// open the Medium Linear Output file for gsvit
	char* mvFname = name;
	//getMediumLinearOutputFilename(boardDoc, filename);
	if(mvFname == NULL)
		return(NULL);
	fprintf(stdout,"medium vector filename: %s\n", mvFname);
	FILE* mvfd = fopen(mvFname, "w");
	if(mvfd == NULL)
	{
		fprintf(stderr, "Unable to open <%s>\n", mvFname);
		return(NULL);
	}
	return(mvfd);
}

void MV_Close(FILE* mvfd)
{
	if(mvfd)
		fclose(mvfd);
}

int MV_ProcessDrillNodeSet(FILE* mvfd, xmlNodeSetPtr xnsPtr, int zstart, int zstop)
{
	int i;
	if((mvfd == NULL) || (xnsPtr == NULL) || (zstart==zstop))
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
		fprintf(stdout, "%d radius:<%s>  \t", i, drillRadius);
		//
		xmlNodeSetPtr xnsPos = XPU_GetNodeSet((xmlDocPtr)currDrill, "./pos/");
		if(xnsPos == NULL)
		{
			fprintf(stdout, "%d radius:<%s>  \t", i, drillRadius);
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
			fprintf(stdout, "%s",(char*)xpos);
			xmlChar* xtype = XPU_LookupFromNode(currPos, "@type");
			fprintf(stdout, "%s",(char*)xtype);
		}
	}
	return(0);
}

