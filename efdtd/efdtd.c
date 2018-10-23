
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
#include "file_processing.h"

// This the main host code for the finite difference
// example.  The kernels are contained in the derivative_m module
#include "finite-difference.h"

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
		retval = FP_ProcessFile(fname, arguments.verbose, arguments.silent);
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
