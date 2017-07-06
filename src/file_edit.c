/*----------------------------------------------------------------------------
*        Headers
*----------------------------------------------------------------------------*/
#define _GNU_SOURCE
#include <stdlib.h>
#include <glib.h>  // for  using GList
#include <unistd.h>	// for fork() and execl
#include <sys/types.h>
#include <sys/wait.h>	// for waitpid
#include <stdio.h>
#include <string.h>
/*----------------------------------------------------------------------------
*        Internal definitions
*----------------------------------------------------------------------------*/
typedef struct s_editRule
{
	char* key;
	char* pRText;
	void (*callback)(struct s_editRule* pthis);
} editRule;

/*----------------------------------------------------------------------------
*        Local variables
*----------------------------------------------------------------------------*/
GList* gRules = NULL;
gint numRules = 0;
/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/
int _copy(char* dest, char* source);

/*----------------------------------------------------------------------------
*        Local functions
*----------------------------------------------------------------------------*/
int _copy(char* dest, char* source)
{
	int childExitStatus;
	pid_t pid;
	int status;
	if (!source || !dest) {
		/* handle as you wish */
		fprintf(stderr, "%s FATAL ERROR NULL args\n",__FUNCTION__);
		return(-1);
	}
	pid = fork();
	if (pid == 0) { /* child */
		execl("/bin/cp", "/bin/cp", source, dest, (char *)0);
	}
	else if (pid < 0) {
		/* error - couldn't start process - you decide how to handle */
		fprintf(stderr, "%s FATAL ERROR couldn't start process\n",__FUNCTION__);
		return(-1);
	}
	else {
		/* parent - wait for child - this has all error handling, you
		* could just call wait() as long as you are only expecting to
		* have one child process at a time.
		*/
		pid_t ws = waitpid( pid, &childExitStatus, WNOHANG);
		if (ws == -1)
		{ /* error - handle as you wish */
			fprintf(stderr, "%s FATAL ERROR unspecified\n",__FUNCTION__);
			return(-1);
		}
		if( WIFEXITED(childExitStatus)) /* exit code in childExitStatus */
		{
			status = WEXITSTATUS(childExitStatus); /* zero is normal exit */
			return(status);
			/* handle non-zero as you wish */
		}
		else if (WIFSIGNALED(childExitStatus)) /* killed */
		{
			fprintf(stderr, "%s killed\n",__FUNCTION__);
			return(-1);
		}
		else if (WIFSTOPPED(childExitStatus)) /* stopped */
		{
			fprintf(stderr, "%s stopped\n",__FUNCTION__);
			return(-1);
		}
	}
	return(0);
}


/*----------------------------------------------------------------------------
*        Exported functions
*----------------------------------------------------------------------------*/

int FE_AppendRule(char* key, char* replacementText, void (*callback)(editRule*) )
{
	editRule* pRule = (editRule*)malloc(sizeof( editRule));
	if( (key==NULL) || (replacementText==NULL) || (pRule ==NULL) )
	{
		fprintf(stderr,"%s Fatal NULL PTR ERROR\n", __FUNCTION__);
		return(-1);
	}
	pRule->key = (char*)malloc( strlen(key)+1);
	pRule->pRText = (char*)malloc(strlen(replacementText) +1);
	pRule->callback = callback;
	if( (pRule->key==NULL) || (pRule->pRText==NULL) )
	{
		fprintf(stderr,"%s Fatal MALLOC ERROR\n", __FUNCTION__);
		return(-1);
	}
	strcpy(pRule->key, key);
	strcpy(pRule->pRText, replacementText);
	gRules = g_list_prepend(gRules, pRule);
	return(0);
}

int FE_Free(void)
{
	//gRules = g_list_reverse(gLayers);
	//gint depth =  g_list_length(gRules);
	GList* l;

	for (l = gRules; l != NULL; l = l->next)
	{	// do something with l->data
		editRule* pRule=  ((editRule*)(l->data));
		free(pRule->key);
		free(pRule->pRText);
		free(pRule);
	}
	return(0);
}

int FE_EditFile(char* filename)
{
	if(numRules == 0)
	{
		gRules = g_list_reverse(gRules);
		numRules =  g_list_length(gRules);
	}
	char tempFname[] = "/tmp/pcb2gsvit_file_edit";
	// copy file to temparary location 
	int retval = _copy(tempFname, filename);
	if( retval != 0)
	{
		fprintf(stderr,"%s ERROR copy %d", __FUNCTION__, retval);
		return(-1);
	}
	
	// parse the temp file looking for keywords
	// if callback is NULL, use default callback
	GList* l;

	for (l = gRules; l != NULL; l = l->next)
	{	// do something with l->data
		editRule* pRule=  ((editRule*)(l->data));
	}	

	return(0);
}


