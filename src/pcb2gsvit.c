#define _GNU_SOURCE
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
// http://www.xmlsoft.org/examples/xpath2.c

#define MAX_FILENAME 0x200

#define SVN_REV "found on github at https://github.com/mpcrowe/pcb2gsvit.git"

int execute_conversion(const char* filename);


int execute_conversion(const char* filename)
{
	xmlDocPtr doc;
	xmlXPathContextPtr xpathCtx;
//	xmlXPathObjectPtr xpathObj;


	// Load XML document
	doc = xmlParseFile(filename);
	if (doc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", filename);
		return(-1);
	}

	// Create xpath evaluation context
	xpathCtx = xmlXPathNewContext(doc);
	if(xpathCtx == NULL)
	{
		fprintf(stderr,"Error: unable to create new XPath context\n");
		xmlFreeDoc(doc);
		return(-1);
	}

	// Register namespaces from list (if any)
//	if((nsList != NULL) && (register_namespaces(xpathCtx, nsList) < 0))
//	{
//		fprintf(stderr,"Error: failed to register namespaces list \"%s\"\n", nsList);
//		xmlXPathFreeContext(xpathCtx);
//		xmlFreeDoc(doc);
//		return(-1);
//	}
	return(0);
}

static void usage(const char *name)
{
	fprintf(stderr, "Usage: %s <xml-file>\n", name);
	fprintf(stderr, "where <xml-file> is a valid description of the pcb to be\n");
	fprintf(stderr, "analyzed, see https://github.com/mpcrowe/pcb2gsvit.git for how to use\n");
}

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

