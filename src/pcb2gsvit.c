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
	xmlXPathFreeContext(xpathCtx);
	if(xpathObj == NULL)
	{
		fprintf(stderr,"Error: unable to evaluate xpath expression \"%s\"\n", xpathString);
		return(NULL);
	}
	if(xmlXPathNodeSetIsEmpty(xpathObj->nodesetval))
	{
		xmlXPathFreeObject(xpathObj);
		fprintf(stderr,"No result\n");
		return(NULL);
	}
	
	xmlNodeSetPtr nodes = xpathObj->nodesetval;

	for(i = 0; i < nodes->nodeNr; ++i)
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
					return(xmlNodeListGetString(doc, nodes->nodeTab[i]->xmlChildrenNode, 1));
				}
			}
		}
		else
		{
			cur = nodes->nodeTab[i];
		}
	}
	return(NULL);
}


#define XPATH_XEM_NAME "//boardInformation/nelmaExport"
#define XPATH_XEM_OUTPUT_FILENAME "//boardInformation/gsvit/mediumLinearFilename"

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

char* getFilename(xmlDocPtr doc, const char* parentDocName, char* dest, const char* xpath)
{
	xmlChar* keyword = xpathSimpleLookup(doc, (char*)xpath);
	if( keyword == NULL)
		return( NULL);
	char* cwd = getFilenamePath(parentDocName);
	if(cwd == NULL)
		return(NULL);
	sprintf(dest, "%s/%s",cwd, keyword );
	xmlFree(keyword);
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

	// Load XML document
	boardDoc = xmlParseFile(filename);
	if (boardDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", filename);
		return(-1);
	}

	// Register namespaces from list (if any)
	//	if((nsList != NULL) && (register_namespaces(xpathCtx, nsList) < 0))
	//	{
	//		fprintf(stderr,"Error: failed to register namespaces list \"%s\"\n", nsList);
	//		xmlXPathFreeContext(xpathCtx);
	//		xmlFreeDoc(boardDoc);
	//		return(-1);
	//	}

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
	xmlChar* cWidth = xpathSimpleLookup(nelmaDoc,XPATH_NELMA_WIDTH);
	if(cWidth == NULL)
		goto processingFault;
	uint32_t width = strtol((char*)cWidth,NULL,10);
	xmlFree(cWidth);

	// get height, in voxels (pixels)
	xmlChar* cHeight = xpathSimpleLookup(nelmaDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
	uint32_t height = strtol((char*)cHeight,NULL,10);
	xmlFree(cHeight);
	fprintf(stdout,"w:%d: h:%d:\n",width, height);
	
	char* mlFname = getMediumLinearOutputFilename(boardDoc, filename);
	if(mlFname == NULL)
		goto processingFault;
		
	fprintf(stdout,"medium linear filename: %s\n", mlFname);

	FILE* mlfd = fopen(mlFname, "w");
	if(mlfd == NULL)
	{
		fprintf(stderr, "unable to open <%s>\n", mlFname);
		goto processingFault;
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

