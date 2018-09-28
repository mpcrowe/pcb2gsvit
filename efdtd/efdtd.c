
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <stdlib.h>
#include <argp.h>
#include <libxml/xmlreader.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>


// This the main host code for the finite difference 
// example.  The kernels are contained in the derivative_m module
#include "finite-difference.h"

static void processNode(xmlTextReaderPtr reader)
{
	// handling of a node in the tree 
	 xmlChar *name, *value;

	name = xmlTextReaderName(reader);
	if (name == NULL)
		name = xmlStrdup(BAD_CAST "--");

	value = xmlTextReaderValue(reader);

	switch( xmlTextReaderNodeType(reader))
	{
	case XML_READER_TYPE_NONE:// = 0,
		printf("none\n");
	break;
	case XML_READER_TYPE_ELEMENT:// = 1,
		printf("start element %s\n", name);
	break;
	case XML_READER_TYPE_ATTRIBUTE:// = 2,
	break;
	case XML_READER_TYPE_TEXT:// = 3,
		printf("text %s\n", name);
	break;
	case XML_READER_TYPE_CDATA:// = 4,
		printf("cdata\n");
	break;
	case XML_READER_TYPE_ENTITY_REFERENCE:// = 5,
		printf("entity reference\n");
	break;
	case XML_READER_TYPE_ENTITY:// = 6,
		printf("entity\n");
	break;
	case XML_READER_TYPE_PROCESSING_INSTRUCTION:// = 7,
		printf("processing instruction\n");
	break;
	case XML_READER_TYPE_COMMENT:// = 8,
		printf("comment\n");
	break;
	case XML_READER_TYPE_DOCUMENT:// = 9,
		printf("document\n");
	break;
	case XML_READER_TYPE_DOCUMENT_TYPE:// = 10,
	break;
	case XML_READER_TYPE_DOCUMENT_FRAGMENT:// = 11,
	break;
	case XML_READER_TYPE_NOTATION:// = 12,
	break;
	case XML_READER_TYPE_WHITESPACE:// = 13,
	break;
	case XML_READER_TYPE_SIGNIFICANT_WHITESPACE:// = 14,
//		printf("whitespace\n");
	break;
	case XML_READER_TYPE_END_ELEMENT:// = 15,
		printf("end element %s\n", name);
	break;
	case XML_READER_TYPE_END_ENTITY:// = 16,
	break;
	case XML_READER_TYPE_XML_DECLARATION:// = 17
	break;
	default:
	break;
	}

//	printf("%d %d %s %d", xmlTextReaderDepth(reader), xmlTextReaderNodeType(reader), name,	xmlTextReaderIsEmptyElement(reader));
	xmlFree(name);
	if (value == NULL)
	{
//		printf("\n");
	}
	else
	{
//		printf(" \"%s\"\n", value);
		xmlFree(value);
	}
}

// main processing of xml files here
static error_t processFile(char* fname, int verbose, int silent)
{
	error_t retval = 0;
//	xmlDocPtr boardDoc;
	int ret;
//        xmlDocPtr xemDoc;
//        char* xemFilename;
//        int i;
//        int j;
//        int k;
	// Load XML document
//        boardDoc = xmlParseFile(fname);
//        if (boardDoc == NULL)
//        {
//                fprintf(stderr, "Error: unable to parse File \"%s\"\n", fname);
//                return(-1);
//        }
//	return(retval);
	xmlTextReaderPtr reader = xmlNewTextReaderFilename(fname);
	if (reader != NULL)
	{
		ret = xmlTextReaderRead(reader);
		while(ret == 1)
		{
			processNode(reader);
			ret = xmlTextReaderRead(reader);
		}
		xmlFreeTextReader(reader);
		if (ret != 0)
		{
			if(!silent)
				printf("%s : failed to parse\n", fname);
			retval = -2;
		}
	}
	else
	{
		if(!silent)
			printf("Unable to open %s\n", fname);
		retval = -1;
	}
	return(retval);
}


// command line options
static struct argp_option options[] = {
	{"verbose",  'v', 0,      0,  "Produce verbose output" },
	{"quiet",    'q', 0,      0,  "Don't produce any output" },
	{"silent",   's', 0,      OPTION_ALIAS },
	{"output",   'o', "FILE", 0,
	"Output to FILE instead of standard output" },
	{ 0 }
};

#define MAX_ARGS 128
// Used by main to communicate with parse_opt. 
struct arguments
{
	char* args[MAX_ARGS];
	int silent;
	int verbose;
	char* output_file;
};


// Parse a single option.
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	/* Get the input argument from argp_parse, which we
	know is a pointer to our arguments structure. */
	struct arguments *arguments = state->input;

	switch (key)
	{
	case 'q': case 's':
		arguments->silent = 1;
	break;
	case 'v':
		arguments->verbose = 1;
	break;
	case 'o':
		arguments->output_file = arg;
	break;

	case ARGP_KEY_ARG:
		if (state->arg_num >= MAX_ARGS)
		/* Too many arguments. */
		argp_usage (state);

		arguments->args[state->arg_num] = arg;

	break;

	case ARGP_KEY_END:
		if (state->arg_num < 1)
		/* Not enough arguments. */
		argp_usage (state);
	break;

	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

// Program command line documentation. 
static char doc[] = "efdtd -- a program to simulate printed circuit boards using FDTD method";
// A description of the arguments we accept. 
static char args_doc[] = "baseline.xml other.xml, ...";

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char* argv[])
{
	struct cudaDeviceProp prop;
	int i;
	int retval = 0;
	struct arguments arguments;

	// Default values for command line arguments
	arguments.silent = 0;
	arguments.verbose = 0;
	arguments.output_file = "-";
	for(i=0;i<MAX_ARGS;i++)
		arguments.args[i] = NULL;

	// Parse our arguments; every option seen by parse_opt will be reflected in arguments. 
	argp_parse(&argp, argc, argv, 0, 0, &arguments);

    
	// get and display GPU device information
	cudaGetDeviceProperties(&prop, 0);
	printf("\nDevice Name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
	if( arguments.verbose > 0)
	{
		printf("Warp size: %d Threads per block: %d\n", prop.warpSize, prop.maxThreadsPerBlock);
		printf("Cpu count: %d Max kernels: %d\n", prop.multiProcessorCount, prop.concurrentKernels);
		printf("Const Mem: %ld(Bytes) Global Mem: %ld(MB)  Shared Mem Per CPU: %ld(Bytes) \n", prop.totalConstMem, prop.totalGlobalMem/1000000, prop.sharedMemPerMultiprocessor);
		printf("Clock Rate: %d(MHz)  Memory Clock rate: %d(MHz)\n\n", prop.clockRate/1000, prop.memoryClockRate/1000);
	}

	/*
	* this initialize the library and check potential ABI mismatches
	* between the version it was compiled for and the actual shared
	* library used.
	*/
	LIBXML_TEST_VERSION
	// Init libxml
	xmlInitParser();

	// main processing loop
	for(i=0;i<MAX_ARGS; i++)
	{
		char* fname = arguments.args[i];
		if(fname == NULL)
			break;
		retval = processFile(fname, arguments.verbose, arguments.silent);
		if(retval)
			break;
	}

	
	// shutdown tasks here
	
	xmlCleanupParser();
	// this is to debug memory for regression tests
	//      xmlMemoryDump();
	if(retval != 0)
	{
		return(retval);
	}	
	setDerivativeParameters(); // initialize 
	dim3 size = {100,100,100};
	SimulationSpace_Create(&size);
	SimulationSpace_Timestep();

	SimulationSpace_Destroy();
	setDerivativeParameters(); // initialize 

	runTest(0); // x derivative
	runTest(1); // y derivative
	runTest(2); // z derivative

	return(0);
}
