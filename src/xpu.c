#include <stdio.h>
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

#include "xpu.h"

xmlChar* XPU_SimpleLookup(xmlDocPtr doc, char* xpathString)
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


xmlChar* XPU_LookupFromNode(xmlNodePtr node, char* xpathString)
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

xmlNodeSetPtr XPU_GetNodeSetFromNode(xmlNodePtr node, char* xpathString)
{
	xmlXPathContextPtr xpathCtx = xmlXPathNewContext( node->doc);
	xmlXPathObjectPtr xpathObj;

	if(xpathCtx == NULL)
	{
		fprintf(stderr,"%s Error: unable to create new XPath context\n", __FUNCTION__);
		return(NULL);
	}
	xpathCtx->node = node;
	xpathObj = xmlXPathEvalExpression((const xmlChar*)xpathString, xpathCtx);
	xmlXPathFreeContext(xpathCtx);
	if(xpathObj == NULL)
	{
		fprintf(stderr,"%s Error: unable to evaluate xpath expression \"%s\"\n", __FUNCTION__, xpathString);
		return(NULL);
	}
	if(xmlXPathNodeSetIsEmpty(xpathObj->nodesetval))
	{
		xmlXPathFreeObject(xpathObj);
		fprintf(stderr,"%s No result\n", __FUNCTION__);
		return(NULL);
	}

	xmlNodeSetPtr nodes = xpathObj->nodesetval;
	return(nodes);
}


xmlNodeSetPtr XPU_GetNodeSet(xmlDocPtr doc, char* xpathString)
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
	return(nodes);
}


double XPU_GetDouble(xmlDocPtr doc, char* xpath)
{
	xmlChar* cVal = XPU_SimpleLookup(doc,xpath);
	if(cVal == NULL)
		return(NAN);

//	fprintf(stderr,"search result <%s>\n",cVal);
	double retval = strtod((char*)cVal, NULL);
	xmlFree(cVal);
	return(retval);
}

