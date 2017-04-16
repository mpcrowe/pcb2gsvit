// frect is a collection of functions to work on a rectangle
// these rectangle layers are stacked to form a 3-object
// member access is through ptr->data[x][y]
// a layer represents a material(s) property, such as permittivity, conductivity
// use separate 3d spaces of layers for each property
// this saves space when for example the relative permeability is always the same
// (for a non-magnetic sytem)


#include <stdio.h>
#include <string.h>
#include "frect.h"

// allocates memory for a new frect instance
fRect* FRECT_New(gint xres, gint yres, gdouble xreal, gdouble yreal, gboolean nullme)
{
	gint i;
	fRect *dc = (fRect *)g_malloc(sizeof(fRect));

	dc->xres = xres;
	dc->yres = yres;
	dc->xreal = xreal;
	dc->yreal = yreal;
	dc->data= (gfloat **) g_malloc(xres*sizeof(gfloat*));
	for (i = 0; i < xres; i++)
	{
		dc->data[i]= (gfloat* )g_malloc(yres*sizeof(gfloat));
	}
	if (nullme) FRECT_Fill(dc, 0);
		return(dc);
}

// allocate memory for a new rectangle of the same
// dimensions of an existing one
fRect* FRECT_NewAlike(fRect *frect, gboolean nullme)
{
	return FRECT_New(frect->xres, frect->yres, frect->xreal, frect->yreal, nullme);
}

// copies a rectangle from one to another (dimensions must be the same)
fRect* FRECT_Copy(fRect* dest, fRect* src)
{
	int i;
	dest->xres = src->xres;
	dest->yres = src->yres;
	dest->xreal = src->xreal;
	dest->yreal = src->yreal;
	for (i = 0; i < src->xres; i++)
	{
		memcpy(dest->data[i] ,src->data[i], src->yres);
	}
	return(dest);
}

// create a new instance of a frect that is a copy of the first
fRect* FRECT_Clone(fRect* src)
{
	fRect* retval = FRECT_NewAlike(src, 0);
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


void FRECT_Fill(fRect *frect, gfloat value)
{
	gint i, j;
	fprintf(stdout,"%s ready to fill\n", __FUNCTION__);
	for (i = 0; i < frect->xres; i++)
	{
		for (j = 0; j < frect->yres; j++)
		{
			frect->data[i][j] = value;
		}
	}
}


void FRECT_Add(fRect *frect, gfloat value)
{
	gint i, j;

	for (i = 0; i < frect->xres; i++)
	{
		for (j = 0; j < frect->yres; j++)
		{
			frect->data[i][j] += value;
		}
	}
}


void FRECT_Scale(fRect *frect, gfloat value)
{
	gint i, j;

	for (i = 0; i < frect->xres; i++)
	{
		for (j = 0; j < frect->yres; j++)
		{
			frect->data[i][j] *= value;
		}
	}
}


