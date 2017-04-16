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
		fprintf(stderr,"%s Error No result for xpath %s\n", __FUNCTION__, xpathString);
		return(NULL);
	}
	
	xmlNodeSetPtr nodes = xpathObj->nodesetval;

	for(i = 0; i < nodes->nodeNr; ++i)
	{
//		fprintf(stdout, "i: %d\n",i);
		if(nodes->nodeTab[i]->type == XML_NAMESPACE_DECL)
		{
			fprintf(stderr, "namespace dec\n");
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
//			fprintf(stdout, "element node\n");
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
		else if( nodes->nodeTab[i]->type == XML_ATTRIBUTE_NODE)
		{
//			fprintf(stderr, "attr node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else if( nodes->nodeTab[i]->type == XML_TEXT_NODE)
		{
//			fprintf(stderr, "text node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else
		{
			fprintf(stderr, "unknown node type %d\n", (nodes->nodeTab[i]->type));
			cur = nodes->nodeTab[i];
		}
	}
	return(NULL);
}


xmlChar* xpathLookupFromNode(xmlNodePtr node, char* xpathString)
{
	xmlXPathContextPtr xpathCtx = xmlXPathNewContext( node->doc);
	xmlXPathObjectPtr xpathObj;
	xmlNodePtr cur;
	int i;

	xpathCtx->node = node;

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
//		fprintf(stderr,"No result for xpath <%s>\n", xpathString);
		return(NULL);
	}
	
	xmlNodeSetPtr nodes = xpathObj->nodesetval;

	for(i = 0; i < nodes->nodeNr; ++i)
	{
//		fprintf(stdout, "i: %d\n",i);
		if(nodes->nodeTab[i]->type == XML_NAMESPACE_DECL)
		{
			fprintf(stderr, "namespace dec\n");
			xmlNsPtr ns;
			ns = (xmlNsPtr)nodes->nodeTab[i];
			cur = (xmlNodePtr)ns->next;
		}
		else if(nodes->nodeTab[i]->type == XML_ELEMENT_NODE)
		{
			fprintf(stdout, "element node\n");
			cur = nodes->nodeTab[i];
			if(cur->ns)
			{
			}
			else
			{
				if(cur->children !=NULL)
				{
					return(xmlNodeListGetString(node->doc, nodes->nodeTab[i]->xmlChildrenNode, 1));
				}
			}
		}
		else if( nodes->nodeTab[i]->type == XML_ATTRIBUTE_NODE)
		{
//			fprintf(stderr, "attr node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else if( nodes->nodeTab[i]->type == XML_TEXT_NODE)
		{
//			fprintf(stderr, "text node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else
		{
			fprintf(stderr, "unknown node type %d\n", (nodes->nodeTab[i]->type));
			cur = nodes->nodeTab[i];
		}
	}
	return(NULL);
}



xmlNodeSetPtr xpathList(xmlDocPtr doc, char* xpathString)
{
	xmlXPathContextPtr xpathCtx;
	xmlXPathObjectPtr xpathObj;
//	xmlNodePtr cur;
//	int i;

	// Create xpath evaluation context
	xpathCtx = xmlXPathNewContext(doc);
	if(xpathCtx == NULL)
	{
		fprintf(stderr,"%s Error: unable to create new XPath context\n", __FUNCTION__);
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
		fprintf(stderr,"%sError: unable to evaluate xpath expression \"%s\"\n", __FUNCTION__, xpathString);
		return(NULL);
	}
	if(xmlXPathNodeSetIsEmpty(xpathObj->nodesetval))
	{
		xmlXPathFreeObject(xpathObj);
		fprintf(stderr,"%s No result\n", __FUNCTION__);
		return(NULL);
	}
	
	xmlNodeSetPtr nodes = xpathObj->nodesetval;
#if 0
	for(i = 0; i < nodes->nodeNr; ++i)
	{
		fprintf(stdout, "i: %d\n",i);
		if(nodes->nodeTab[i]->type == XML_NAMESPACE_DECL)
		{
			fprintf(stderr, "namespace dec\n");
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
			fprintf(stdout, "element node\n");
			cur = nodes->nodeTab[i];
			if(cur->ns)
			{
			}
			else
			{
				if(cur->children !=NULL)
				{
//					return(xmlNodeListGetString(doc, nodes->nodeTab[i]->xmlChildrenNode, 1));
				}
			}
		}
		else if( nodes->nodeTab[i]->type == XML_ATTRIBUTE_NODE)
		{
			fprintf(stderr, "attr node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
//			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else if( nodes->nodeTab[i]->type == XML_TEXT_NODE)
		{
			fprintf(stderr, "text node <%s>\n", xmlNodeGetContent(nodes->nodeTab[i]));
//			return(xmlNodeGetContent(nodes->nodeTab[i]));
		}
		else
		{
			fprintf(stderr, "unknown node type %d\n", (nodes->nodeTab[i]->type));
			cur = nodes->nodeTab[i];
		}
	}
#endif
	return(nodes);
}


#define XPATH_XEM_NAME "/boardInformation/nelmaExport/text()"
#define XPATH_XEM_OUTPUT_FILENAME "/boardInformation/gsvit/mediumLinearFilename/text()"
#define XPATH_XEM_MATERIALS "/boardInformation/materials/material"
#define XPATH_XEM_LAYERS "/boardInformation/boardStackup/layer"


#define XPATH_NELMA_WIDTH "/nelma/space/width/text()"
#define XPATH_NELMA_HEIGHT "/nelma/space/height/text()"
#define XPATH_NELMA_RES	"/nelma/space/resolution/text()"
#define XPATH_NELMA_RES_UNITS	"/nelma/space/resolution/@units"
        		                		
					
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

double getXpathDouble(xmlDocPtr doc, char* xpath)
{
	// get width, in voxels (pixels)
	xmlChar* cVal = xpathSimpleLookup(doc,xpath);
	if(cVal == NULL)
		return(NAN);
		
//	fprintf(stderr,"search result <%s>\n",cVal);
	double retval = strtod((char*)cVal, NULL);
	xmlFree(cVal);
	return(retval);
}

void getXpathTest(xmlDocPtr doc, char* xpath)
{
	// get width, in voxels (pixels)
	fprintf(stderr,"%s looking for <%s>\n",__FUNCTION__, xpath);
	xmlChar* cVal = xpathSimpleLookup(doc,xpath);
	if(cVal == NULL)
		return;
		
	fprintf(stderr,"search result <%s>\n",cVal);
	xmlFree(cVal);
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

	// get the pixel resolution, in meters per pixel
	double res = getXpathDouble(nelmaDoc, XPATH_NELMA_RES);
	if(isnan(res))
		goto processingFault;	
	xmlChar* units = xpathSimpleLookup(nelmaDoc, XPATH_NELMA_RES_UNITS);
	res = scaleToMeters(res, (char*)units);
	fprintf(stdout, "adjusted res: %g\n", res);
	
//	xmlNodeSetPtr xnsMaterials  = xpathList(boardDoc, XPATH_XEM_MATERIALS);
//	if(xnsMaterials == NULL)
//		goto processingFault;

	xmlNodeSetPtr xnsLayers  = xpathList(boardDoc, XPATH_XEM_LAYERS);
	if(xnsLayers == NULL)
		goto processingFault;
	
	
	fRect* fillLayerEr = FRECT_New(width, height, res, res, 0);
	FRECT_Fill(fillLayerEr, 1.0);
//	PgFRect* mu = sv_fcube_new_alike(fillLayerEr, 1);
//	PgFRect* sigma = sv_fcube_new_alike(fillLayerEr, 1);
//	PgFRect* sigast = sv_fcube_new_alike(fillLayerEr, 1);
                                                      
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
		
		xmlChar* layerName = xpathLookupFromNode(currLayer, "./name/text()");
		fprintf(stdout, "%d name:<%s>  \t", i, layerName);
		xmlChar* materialName = xpathLookupFromNode(currLayer, "./material/text()");
		if(materialName == NULL)
		{
			fprintf(stderr, "\nError, no material name specified\n");
			fclose(mlfd);
			goto processingFault;
		}
		fprintf(stdout, "material:<%s> \t",  materialName);
			
				                
		
		xmlChar* cThickness = xpathLookupFromNode(currLayer, "./thickness/text()");
		if(cThickness != NULL)
		{
			fprintf(stdout, "thickness:<%s>", cThickness);
		}
		fprintf(stdout, "\n");

		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/relativePermittivity/text()", materialName);
		xmlChar* cEr = xpathSimpleLookup(boardDoc, xpathString);
		if(cEr == NULL)  
		{
			fclose(mlfd);
			goto processingFault;
		}
		fprintf(stdout, "Er: %s\n",  cEr);

		sprintf(xpathString, "/boardInformation/materials/material[@id='%s']/conductivity/text()", materialName);
		xmlChar* cCond = xpathSimpleLookup(boardDoc, xpathString);
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

