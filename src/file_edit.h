#ifndef _FILE_EDIT
#define FILE_EDIT 1
/*----------------------------------------------------------------------------
*        Headers
*----------------------------------------------------------------------------*/
#include <stdlib.h>
#include <sys/types.h>
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
*        Exported functions
*----------------------------------------------------------------------------*/
int FE_AppendRule(char* key, char* replacementText, void (*callback)(editRule*) );
int FE_Free(void);

int FE_EditFile(char* filename);
#endif
