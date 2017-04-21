#ifndef _XPU
#define _XPU 1

//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//#include <stdint.h>
//#include <stddef.h>
#include <math.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <unistd.h>


xmlChar* XPU_SimpleLookup(xmlDocPtr doc, char* xpathString);

xmlChar* XPU_LookupFromNode(xmlNodePtr node, char* xpathString);

xmlNodeSetPtr XPU_GetNodeSet(xmlDocPtr doc, char* xpathString);

double XPU_GetDouble(xmlDocPtr doc, char* xpath);

#endif
