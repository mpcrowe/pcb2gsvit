// frect is a collection of functions to work on a rectangle
// these rectangle layers are stacked to form a 3-object
// member access is through ptr->data[x][y]


#include <stdio.h>
#include <string.h>
#include "frect.h"

// allocates memory for a new frect instance
fRect* FRECT_New(gint xres, gint yres)
{
	gint i;
	fRect *dc = (fRect *)g_malloc(sizeof(fRect));

	dc->xres = xres;
	dc->yres = yres;
	dc->data= (indexSize_t **) g_malloc(xres*sizeof(indexSize_t*));
	for (i = 0; i < xres; i++)
	{
		dc->data[i]= (indexSize_t* )g_malloc(yres*sizeof(indexSize_t));
	}
	return(dc);
}

// allocate memory for a new rectangle of the same
// dimensions of an existing one
fRect* FRECT_NewAlike(fRect *frect)
{
	return FRECT_New(frect->xres, frect->yres);
}

// copies a rectangle from one to another (dimensions must be the same)
fRect* FRECT_Copy(fRect* dest, fRect* src)
{
	int i;
	dest->xres = src->xres;
	dest->yres = src->yres;
	for (i = 0; i < src->xres; i++)
	{
		memcpy(dest->data[i] ,src->data[i], src->yres*sizeof(indexSize_t));
	}
	return(dest);
}

// create a new instance of a frect that is a copy of the first
fRect* FRECT_Clone(fRect* src)
{
	fRect* retval = FRECT_NewAlike(src);
	return(FRECT_Copy(retval, src));
}

// deconstructor
void FRECT_Free(fRect *frect)
{
	gint i;
	for (i = 0; i < frect->xres; i++)
	{
		g_free((void *) frect->data[i]);
	}

	g_free((void *)frect->data);
	frect->data=NULL;
}


void FRECT_Fill(fRect *frect, indexSize_t index)
{
	gint i, j;
	fprintf(stdout,"%s ready to fill\n", __FUNCTION__);
	for (i = 0; i < frect->xres; i++)
	{
		for (j = 0; j < frect->yres; j++)
		{
			frect->data[i][j] = index;
		}
	}
}
   
