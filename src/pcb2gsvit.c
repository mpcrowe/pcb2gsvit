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

#define SVN_REV "$Id: gcdcQCReporter.c 138 2014-04-28 23:29:34Z  $"

//node for a linked list
struct node{
	char data[100];	// will store information
	int index;
	int lineNumber;
	struct node *next;	// the reference to the next node
};


struct valueNode
{
	char* name;
	float err_min;
	float err_max;
	float oof_min;
	float oof_max;
	float mean;
	float oof_deviation;
	float err_deviation;
	int index;
	int lineNumber;
	char* err_string;
	char* oof_string;
	int err_code;

	struct valueNode* next;
};



static int verbose_flag = 0;
static int version_flag = 0;
static int nooof_flag = 0;
static int cont_flag = 0;
static int useBigIndex_flag = 0;

struct valueNode* boundsValueList = NULL;
struct node *list;

void dumpValueNode(struct valueNode* ptr, char*pHeader);
void recalcBounds(struct valueNode* pInstance);


void recalcBounds(struct valueNode* pInstance)
{
	struct valueNode* ptr = pInstance;
	if(ptr->err_deviation > 0.0)
	{
		ptr->err_max = ptr->mean + ptr->err_deviation;
		ptr->err_min = ptr->mean - ptr->err_deviation;
	}

	if(ptr->oof_deviation > 0.0)
	{
		ptr->oof_max = ptr->mean + ptr->oof_deviation;
		ptr->oof_min = ptr->mean - ptr->oof_deviation;
	}
//	dumpValueNode(pInstance, "recalc");
	
}

void usage()
{
	printf("\t --text         input the source text file that contains the test information\n");
	printf("\t --xml          input the source xml file that contains the testing standards\n");
	printf("\t --nooof        suppresses warnings for out of family data\n");
	printf("\t --cont         continue in the event of a failed test\n");
	printf("\t --verbose      enables diagnostic printouts\n");
	printf("\t --help         this iexits help info\n");
}


void dumpValueNode(struct valueNode* ptr, char*pHeader)
{
	printf("%s %s",__FUNCTION__, pHeader);
	if(ptr == NULL) { printf("NULL POINTER\n"); return; }

	printf("value Node name: %s\n", ptr->name);
	printf("err_min: %f\n", ptr->err_min);
	printf("err_max: %f\n", ptr->err_max);
	printf("oof_min: %f\n", ptr->oof_min);
	printf("oof_max: %f\n", ptr->oof_max);
	printf("err_string: %s\n", ptr->err_string);
	printf("oof_string: %s\n", ptr->oof_string);
	printf("err_code: %d\n", ptr->err_code);
	printf("index: %d\n", ptr->index);
	if(ptr->lineNumber) printf("lineNumber: %d\n\n",ptr->lineNumber);
	if(ptr->next)
		dumpValueNode(ptr->next, pHeader);
}


void dumpNode(struct node* ptr)
{
	printf("\nstart of node dump\n");
	printf("data: %s\n", ptr->data);
	printf("index: %d\n\n", ptr->index);
}


struct node *newNode(char* data, int lineNumber, int tokenNumber)
{
	int len = 0;
	struct node* retval = (struct node*)malloc(sizeof(struct node));
	strcpy(retval->data, data);
	// remove newline
	len = strlen(retval->data);
	if( retval->data[len-1] == '\n' )
		retval->data[len-1] = 0;
		
	if(useBigIndex_flag == 0)
	{
		char string[3];
		sprintf(string, "%d%d", tokenNumber, lineNumber);
		retval->index = atoi(string);
		retval->lineNumber = 0;
	}
	else
	{
		retval->index = tokenNumber;
		retval->lineNumber = lineNumber;
	}
	retval->next = NULL;

//	if(verbose_flag){ dumpNode(retval); }
	return retval;
}


xmlChar* getTextInNode(xmlTextReaderPtr reader)
{
	xmlChar* value;
	int ret = xmlTextReaderRead(reader);
	while (ret == 1)
	{
		if( xmlTextReaderNodeType(reader)== XML_READER_TYPE_TEXT)
		{
			value = xmlTextReaderValue(reader);
			return(value);
		}
		if(xmlTextReaderNodeType(reader) == XML_READER_TYPE_END_ELEMENT)
			return(NULL);
		ret = xmlTextReaderRead(reader);
	}
	return(NULL);
}


// used recursivly when a range of index values are expanded
struct valueNode* expandValueNode(struct valueNode* parent, int end)
{
	struct valueNode* retval = (struct valueNode*)malloc(sizeof(struct valueNode));
	if(retval == NULL)
		return(retval);
	memset(retval, 0, sizeof(struct valueNode));
	memcpy(retval, parent, sizeof(struct valueNode));
	retval->next = NULL;
	retval->index = parent->index+1;
	if(verbose_flag){ printf("%s index %d\n", __FUNCTION__, retval->index); }
	if(retval->index < end)
		retval->next = expandValueNode(retval, end);
	return(retval);
}


struct valueNode* copyValueNode(struct valueNode* src, char* pIndexString, char* pLineString)
{
	if(verbose_flag)
	{
		printf("%s index: %s  line %s\n",__FUNCTION__, pIndexString, pLineString); 
	}
	struct valueNode* retval = (struct valueNode*)malloc(sizeof(struct valueNode));
	if(retval == NULL)
		return(retval);
	memset(retval, 0, sizeof(struct valueNode));

	memcpy(retval, src, sizeof(struct valueNode));
	retval->next = NULL;

	if(pLineString)
	{
		retval->lineNumber = atoi(pLineString);
		useBigIndex_flag = 1;
	}
	else
		retval->lineNumber = 0;

	char* range = strstr(pIndexString,"[");
	if(range)
	{	
		int start = atoi(range+1);
		range = strstr(range,":");
		int end = atoi(range+1);
		if(verbose_flag)
		{
			printf("index range: %d to %d\n",start, end);
		}
		retval->index = start;
		retval->next = expandValueNode(retval, end);
	}
	else
		retval->index = atoi(pIndexString);
//	dumpValueNode(retval);
	return(retval);	
}


struct valueNode* newValueNode(xmlTextReaderPtr reader)
{
	struct valueNode tempNode;
	char* pIndexString = NULL;
	char* pLineString = NULL;
	memset(&tempNode, 0, sizeof(struct valueNode));

	while(1)
	{
		xmlTextReaderRead(reader);
		if(xmlTextReaderNodeType(reader) == XML_READER_TYPE_END_ELEMENT)
		{
			char* nodeName = (char*)xmlTextReaderName(reader);

			if( strcmp(nodeName, "value") == 0)
			{
				struct valueNode* retval = copyValueNode(&tempNode, pIndexString, pLineString);
				return( retval) ;
			}
			continue;
		 }
		if(xmlTextReaderNodeType(reader) == XML_READER_TYPE_SIGNIFICANT_WHITESPACE)
			continue;

		char* nodeName = (char*)xmlTextReaderName(reader);
		char* str = (char*)getTextInNode(reader);

		if(verbose_flag){ printf("name: (%s) val: (%s)\n",nodeName, str); }

		float fval = (float)atof(str);
		if(!strcmp(nodeName, "name"))
			tempNode.name = str;
		else if(!strcmp(nodeName, "err-min"))
		{
			tempNode.err_min = fval;
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "err-max"))
		{
			tempNode.err_max = fval;
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "oof-min"))
		{
			tempNode.oof_min = fval;
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "oof-max"))
		{
			tempNode.oof_max = fval;
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "mean"))
		{
			tempNode.mean = fval;
			recalcBounds(&tempNode);
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "oof-deviation"))
		{
			tempNode.oof_deviation = fval;
			recalcBounds(&tempNode);
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "err-deviation"))
		{
			tempNode.err_deviation = fval;
			recalcBounds(&tempNode);
			xmlFree(str);
		}
		else if(!strcmp(nodeName, "err-string"))
			tempNode.err_string = str;
		else if(!strcmp(nodeName, "oof-string"))
			tempNode.oof_string = str;
		else if(!strcmp(nodeName, "continue"))
		{
			tempNode.err_code = atoi(str);
			xmlFree(str);
		}
			
		else if(!strcmp(nodeName, "index"))
			pIndexString = str;
		else if(!strcmp(nodeName, "lineNumber"))
			pLineString = str;
		else if(!strncmp(nodeName, "*", 1))
		{
			
		}
	}  // end while
	return(NULL);
}


int parseBoundsFile(char *filename) 
{ //this method parses the xml document and populates a linked list called boundsValueList
	int ret = 1;								
	struct valueNode* lastNode;
	xmlTextReaderPtr reader = xmlNewTextReaderFilename(filename);
	if(reader == NULL)
	{
		printf("Unable to open <%s>\n", filename);
		return(-1);
	}

	while (ret == 1) 
	{
		ret = xmlTextReaderRead(reader);
		if(xmlTextReaderNodeType(reader) == XML_READER_TYPE_SIGNIFICANT_WHITESPACE) continue;
		if(xmlTextReaderNodeType(reader) == XML_READER_TYPE_END_ELEMENT)
		{
			if(strcmp((char*)xmlTextReaderName(reader), "file") == 0)
			{
				ret = 0;
				break;
			}
			continue;
		}
		if(verbose_flag){ printf("%s\n", (char*)xmlTextReaderName(reader)); }
		if(strcmp((char*)xmlTextReaderName(reader), "value") == 0)
		{
			struct valueNode* retval = NULL;
			retval = newValueNode(reader);
			if(retval != NULL)
			{
				if(boundsValueList == NULL)
				{
					boundsValueList = retval;
					lastNode = boundsValueList;
				}
				else
					lastNode->next = retval;
				while(lastNode->next != NULL)
				{
					lastNode = lastNode->next;
				}
			}
		}
	}  // end while
	xmlFreeTextReader(reader);
	if (ret == -1) 
	{
		printf("%s : failed to parse %d\n", filename, ret);
	}
	if(verbose_flag){ printf("%s retval: %d\n", __FUNCTION__, ret); }
	return(ret);
}


// returns 0 on no error
int testCheck(struct valueNode* boundsList,struct node* txtList)
{
	int err_flag = 0;
	int oof_flag = 0;
	int continue_flag = 0;
	float num = 0;
	int match_flag = 0;
	while(txtList != NULL)
	{
		int xmlIndex = boundsList->index;
		int txtIndex = txtList->index;
//		if(verbose_flag)
//		{
//			printf("\n%s data:  %s\n", __FUNCTION__, txtList->data);
//			printf("%sbounds: %d:%d, txt: %d:%d\n",__FUNCTION__, boundsList->lineNumber,boundsList->index, txtList->lineNumber,txtList->index);
//		}
		if(useBigIndex_flag ==0)
		{
			if(xmlIndex == txtIndex)
			{
				match_flag = 1;
				break;			
			}
		}
		else
		{
			if( (xmlIndex == txtIndex) && (boundsList->lineNumber == txtList->lineNumber) )
			{
				match_flag = 1;
				break;			
			}
		}
		txtList = txtList->next;
	} // end while
	if(match_flag ==0)
	{
		printf("%s Warning, No match found for line %d token %d\n",__FUNCTION__, boundsList->lineNumber, boundsList->index);
		return(0);
	}

	if(match_flag)
	{
		num = (float)atof(txtList->data);
		if(verbose_flag){ printf("%s %s: err_min %f, val %f, err_max %f\n", __FUNCTION__, boundsList->name, boundsList->err_min, num, boundsList->err_max); }
		if(num < boundsList->err_min)
			err_flag = 1;
		if(num > boundsList->err_max)
			err_flag = 1;
		if(num < boundsList->oof_min)
			oof_flag = 1;
		if(num > boundsList->oof_max)
			oof_flag = 1;

		continue_flag = boundsList->err_code;
	}

	if(err_flag != 0)
	{
		printf("%s. ", boundsList->err_string);
		if(useBigIndex_flag) printf("line #:%d, idx:%d\n", boundsList->lineNumber, boundsList->index);
		if(num > boundsList->err_max)
			printf("%s = %f > %f\n", boundsList->name, num, boundsList->err_max);
		else
			printf("%s = %f < %f\n", boundsList->name, num, boundsList->err_min);
		return( continue_flag);
	}

	if((oof_flag != 0) && (nooof_flag == 0))
	{
		printf("%s. ", boundsList->oof_string);
		if(useBigIndex_flag) printf("line #:%d, idx:%d\n", boundsList->lineNumber, boundsList->index);
		if(num > boundsList->oof_max)
			printf("%s = %f > %f\n", boundsList->name, num,boundsList->oof_max);
		else
			printf("%s = %f < %f\n", boundsList->name, num, boundsList->oof_min);
		return( 0);
	}
	return( 0);
}



int main(int argc, char *argv[])
{
	int c;
	list = NULL;
	int returnValue = 0; // THIS IS THE OFFICIAL PROGRAM RETURN VALUE!!!!!!
	char textFileName[MAX_FILENAME];
	char xmlFileName[MAX_FILENAME];

	if(argc <=1)
	{
		usage();
		exit(2);
	}

	//parse command-line options.
	while(1)
	{
		static struct option long_options[] =
		{
			{"verbose", no_argument, &verbose_flag, 1},
			{"version", no_argument, &version_flag, 1},
			{"nooof", no_argument, &nooof_flag, 1},
			{"cont", no_argument, &cont_flag, 1},
			{"text", required_argument, 0, 'a'},
			{"xml", required_argument, 0, 'b'},
			{"help", no_argument, 0, '?'},
			{0, 0, 0, 0}
		};

		int option_index = 0;
		c = getopt_long (argc, argv, "axr:d:f:", long_options, &option_index);
		/* Detect the end of the options. */
		if(c == -1)
			break;

		switch(c)
		{
		case 0:
			if(long_options[option_index].flag != 0)
				break;
		case'a':
			//printf("copying txt");
			strncpy(textFileName, optarg, MAX_FILENAME);
			break;
		case 'b':
			//printf("copying testfile");
			strncpy(xmlFileName, optarg, MAX_FILENAME);
			break;
		case '?':
			usage();
			exit(0);
			break;
		default:
			usage();
			abort();
		}

	} // end while

	if(version_flag)
	{
		printf("Version: %s\n", SVN_REV);
		exit(0);
	}
	if(verbose_flag){ printf("\n\n\nReading and parsing <%s>\n",xmlFileName); }
	//read in test limits 
	if(parseBoundsFile(xmlFileName))
	{
		fprintf(stderr,"ERROR unable to parse bounds file\n");
		exit(3);
	}

	if(verbose_flag){ printf("\n\n\nReading and parsing <%s>\n",textFileName); }
	
	//read and parse file to be checked
	FILE *fp;
	char mode = 'r';
	fp = fopen(textFileName, &mode);
	if(!fp)
	{
		fprintf(stderr,"ERROR, unable to open <%s>\n",textFileName);
		exit(2);
	}
	char* line = NULL;
	int lineCount = 1;
	//node* current = NULL;
	struct node* lastNode = (struct node*)malloc(sizeof(struct node));
	lastNode->next = NULL;
	
	size_t size = 0;
	ssize_t size2 = 0;
	size2 = getline(&line, &size, fp);
	if(size2 < 0)
	{
		fprintf(stderr, "ERROR reading <%s>\n",textFileName);
		fclose(fp);
		exit(2);
	}

	while( size2 > 0 )
	{
		char *str;
		str = strtok( line, ":,");
		//char *delimit;
		int tokenCount = 1;
		while(str)
		{
			struct node* current = NULL;
			current = newNode(str,lineCount,tokenCount);
			if(list == NULL)
				list = current;
			else
				lastNode->next = current;
			lastNode = current;

			if(verbose_flag)
			{
				printf("next:  %s   indexNum: %d    lineNum:%d\n", current->data, tokenCount, lineCount);
			}
			tokenCount++;
			str = strtok( NULL, ":,");

		} // end while
		lineCount++;
		size2 = getline(&line, &size, fp);
	}  // end while, done reading input file
	fclose(fp);
	free(line);

	if( verbose_flag && useBigIndex_flag) 
	{
		printf("\r\n\r\nBounds file must use <line_number>,  big indexes in use\r\n\r\n");
	}


	if(verbose_flag){ printf("\n\n\nStarting bounds checking\n"); }

	// check data file against limits
	while(boundsValueList != NULL)
	{
		int ret = testCheck(boundsValueList, list);
		if(ret != 0)
		{
			returnValue = ret;
			if(!cont_flag) break;
		}
		boundsValueList = boundsValueList->next;
	}

	return(returnValue);
}










