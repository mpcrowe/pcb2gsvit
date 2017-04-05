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

// http://www.xmlsoft.org/examples/xpath2.c

#define MAX_FILENAME 0x200

#define SVN_REV "found on github at https://github.com/mpcrowe/pcb2gsvit.git"

int execute_conversion(const char* filename);
char* getNelmaFilename(xmlDocPtr doc, const char* parentDocName);

xmlChar* xpathSimpleLookup(xmlDocPtr doc, char* xpathString)
{
	xmlXPathContextPtr xpathCtx;
	xmlXPathObjectPtr xpathObj;
	xmlNodePtr cur;
	int size;
	int i;

	// Create xpath evaluation context
	xpathCtx = xmlXPathNewContext(doc);
	if(xpathCtx == NULL)
	{
		fprintf(stderr,"Error: unable to create new XPath context\n");
		return(NULL);
	}

	// Register namespaces from list (if any)
	//	if((nsList != NULL) && (register_namespaces(xpathCtx, nsList) < 0))
	//	{
	//		fprintf(stderr,"Error: failed to register namespaces list \"%s\"\n", nsList);
	//		xmlXPathFreeContext(xpathCtx);
	//		xmlFreeDoc(doc);
	//		return(-1);
	//	}
	// Evaluate xpath expression

	xpathObj = xmlXPathEvalExpression((const xmlChar*)xpathString, xpathCtx);
	if(xpathObj == NULL)
	{
		fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", xpathString);
		xmlXPathFreeContext(xpathCtx);
		return(NULL);
	}
	xmlNodeSetPtr nodes = xpathObj->nodesetval;

	size = (nodes) ? nodes->nodeNr : 0;

	for(i = 0; i < size; ++i)
	{
		if(nodes->nodeTab[i]->type == XML_NAMESPACE_DECL)
		{
			xmlNsPtr ns;
			ns = (xmlNsPtr)nodes->nodeTab[i];
			cur = (xmlNodePtr)ns->next;
//			if(cur->ns)
//			{
//				fprintf(stdout, "= namespace \"%s\"=\"%s\" for node %s:%s\n",
//				ns->prefix, ns->href, cur->ns->href, cur->name);
//			}
//			else
//			{
//				fprintf(stdout, "= namespace \"%s\"=\"%s\" for node %s\n",
//				ns->prefix, ns->href, cur->name);
//			}
		}
		else if(nodes->nodeTab[i]->type == XML_ELEMENT_NODE)
		{
			cur = nodes->nodeTab[i];
			if(cur->ns)
			{
			}
			else
			{
				if(cur->children !=NULL)
				{
					xmlChar* keyword;
					keyword = xmlNodeListGetString(doc, nodes->nodeTab[i]->xmlChildrenNode, 1);
					xmlXPathFreeContext(xpathCtx);
					return(keyword);
				}
			}
		}
		else
		{
			cur = nodes->nodeTab[i];
		}
	}
	xmlXPathFreeContext(xpathCtx);
	return(NULL);
}


#define XPATH_XEM_NAME "//boardInformation/nelmaExport"
#define XPATH_NELMA_WIDTH "//nelma/space/width"
#define XPATH_NELMA_HEIGHT "//nelma/space/height"

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

char* getNelmaFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];

	xmlChar* keyword = xpathSimpleLookup(doc, XPATH_XEM_NAME);
	if( keyword == NULL)
		return( NULL);
	char* cwd = getFilenamePath(parentDocName);
	if(cwd == NULL)
		return(NULL);
	sprintf(fullName, "%s/%s",cwd, keyword );
	xmlFree(keyword);
	return(fullName);
}


int execute_conversion(const char* filename)
{
	xmlDocPtr doc;
	xmlDocPtr nelmaDoc;
	char* nelmaFilename;

	// Load XML document
	doc = xmlParseFile(filename);
	if (doc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", filename);
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

	nelmaFilename = getNelmaFilename(doc, filename);
	if(nelmaFilename == NULL)
	{
		goto processingFault;
	}
	fprintf(stdout, "%s\n",nelmaFilename);
	nelmaDoc = xmlParseFile(nelmaFilename);
	if(nelmaDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", nelmaFilename);
		return(-1);
	}
	xmlChar* cWidth = xpathSimpleLookup(nelmaDoc,XPATH_NELMA_WIDTH);
	if(cWidth == NULL)
		goto processingFault;
	xmlChar* cHeight = xpathSimpleLookup(nelmaDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
		
	fprintf(stderr,"w:%s: h:%s:\n",cWidth, cHeight);
	uint32_t width = strtol((char*)cWidth,NULL,10);
	uint32_t height = strtol((char*)cHeight,NULL,10);
	
	fprintf(stderr,"w:%d: h:%d:\n",width, height);
		

	fprintf(stderr, "processing complete, no errors encountered\n");
	return(0);

processingFault:
	fprintf(stderr, "processing fault\n");
	xmlFreeDoc(doc);
	return(-1);
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

