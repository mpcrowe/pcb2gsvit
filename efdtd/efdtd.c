
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
#include <glib.h>  // for  using GList

#include "../src/xpu.h"
#include "../src/xpathConsts.h"
#include "../src/material.h"


// This the main host code for the finite difference
// example.  The kernels are contained in the derivative_m module
#include "finite-difference.h"

static char* getXemFilename(xmlDocPtr doc, const char* parentDocName)
{
	static char fullName[0x400];
	return(XPU_GetFilename(doc, parentDocName, fullName, XPATH_XEM_NAME) );
}


// main processing of xml files here
static error_t processFile(char* fname, int verbose, int silent)
{
	error_t retval = 0;
	xmlDocPtr boardDoc;
	xmlDocPtr xemDoc;
	char* xemFilename;

	// Load XML document
	boardDoc = xmlParseFile(fname);
	if (boardDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse File \"%s\"\n", fname);
		return(-1);
	}
	
	xmlNodeSetPtr boardDocFrag = XPU_GetNodeSet(boardDoc, XPATH_XEM_BOARD_DOC);
	if(boardDocFrag == NULL)
		goto processingFault;
	
	// create the materials table
	xmlNodeSetPtr xnsMaterials  = XPU_GetNodeSet(boardDoc, XPATH_XEM_MATERIALS);
	if(xnsMaterials == NULL)
		goto processingFault;

	retval = MATRL_CreateTableFromNodeSet(xnsMaterials);
	if(retval)
		goto processingFault;

	MATRL_DumpAll();

       // get xem filename
	xemFilename = getXemFilename(boardDoc, fname);
	if(xemFilename == NULL)
	{
		goto processingFault;
	}
	fprintf(stdout, "%s\n",xemFilename);

	// parse xem file
	xemDoc = xmlParseFile(xemFilename);
	if(xemDoc == NULL)
	{
		fprintf(stderr, "Error: unable to parse file \"%s\"\n", xemFilename);
		return(-1);
	}

	// get width, in voxels (pixels)
	xmlChar* cWidth = XPU_SimpleLookup(xemDoc,XPATH_NELMA_WIDTH);
	if(cWidth == NULL)
		goto processingFault;
	gint width = strtol((char*)cWidth,NULL,10);
	xmlFree(cWidth);

	// get height, in voxels (pixels)
	xmlChar* cHeight = XPU_SimpleLookup(xemDoc,XPATH_NELMA_HEIGHT);
	if(cHeight == NULL)
		goto processingFault;
	gint height = strtol((char*)cHeight,NULL,10);
	xmlFree(cHeight);

	fprintf(stdout,"w:%d: h:%d:\n",width, height);

	// get the pixel resolution, in meters per pixel
	double res = XPU_GetDouble(xemDoc, XPATH_NELMA_RES);
	if(isnan(res))
		goto processingFault;
	xmlChar* units = XPU_SimpleLookup(xemDoc, XPATH_NELMA_RES_UNITS);
	res = MATRL_ScaleToMeters(res, (char*)units);
	fprintf(stdout, "adjusted res: %g\n", res);


	// get the layers from the board document
	xmlNodeSetPtr xnsLayers  = XPU_GetNodeSet(boardDoc, XPATH_XEM_LAYERS);
	if(xnsLayers == NULL)
		goto processingFault;

processingFault:
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
/*	setDerivativeParameters(); // initialize
	dim3 size = {100,100,100};
	SimulationSpace_Create(&size);
	SimulationSpace_Timestep();

	SimulationSpace_Destroy();
	setDerivativeParameters(); // initialize

	runTest(0); // x derivative
	runTest(1); // y derivative
	runTest(2); // z derivative
*/
	return(0);
}
