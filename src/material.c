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

#include <glib.h>

#include "material.h"

material_t* materialTable = NULL;

void MATRL_Init(indexSize_t tableSize)
{
//	int i;
	fprintf(stdout, "%s  %d\n", __FUNCTION__, tableSize);
	materialTable = g_malloc(sizeof(material_t) * tableSize);
//	for(i=0; i<tableSize;i++)
//	{
//		materialTable[i] = g_malloc(sizeof(material_t));
//	}
	
}

        void MATRL_CreateTableFromNodeSet(xmlNodeSetPtr xnsMaterials)
        {
        	fprintf(stdout, "%s\n", __FUNCTION__);
        	
        }
        