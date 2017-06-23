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

// here we use SV_MAT_LINEAR see gsvit/src3d/settings.h line 102 and plan.c line 245
// 		typ epslsonR, conductivity, muR, magnetic conductivity
#define COPPER_CYL_INFO "0 1.0 5.96e7 1.0 0.0"
#define AIR_CYL_INFO    "0 1.0 0.0 1.0 0.0"


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

int MV_ProcessDrillNodeSet(FILE* mvfd, xmlNodeSetPtr xnsPtr, int z1, int z2, int plateThickness)
{
	int i;
	int isPlated = 1;
	if((mvfd == NULL) || (xnsPtr == NULL) || (z1==z2))
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
				fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius, COPPER_CYL_INFO);
				if((radius-plateThickness) > 0)
					fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius-plateThickness, AIR_CYL_INFO);
			}
			else
			{ // an unplaed hole consists of a cylinder of air
				fprintf(mvfd,"7 %d %d %d %d %d %d %d %s\n", xp,yp,z1,xp,yp,z2,radius, AIR_CYL_INFO);
			}
			
			        			
		}
	}
	return(0);
}

