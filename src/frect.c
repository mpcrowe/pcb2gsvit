
/*
 *  Copyright (C) 2013 Petr Klapetek
 *  E-mail: klapetek@gwyddion.net.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111 USA
 */

/*  fcube.c : 
 *  floating point 3D data representation
 *
 */

#include <stdio.h>
#include "frect.h"

PgFRect* PGFrectNew(gint xres, gint yres, gdouble xreal, gdouble yreal, gboolean nullme)
{
    gint i;
    PgFRect *dc = (PgFRect *)g_malloc(sizeof(PgFRect));

    dc->xres = xres;
    dc->yres = yres; 
    dc->xreal = xreal;
    dc->yreal = yreal;
    dc->data= (gfloat **) g_malloc(xres*sizeof(gfloat*));
    for (i = 0; i < xres; i++)
    {
        dc->data[i]= (gfloat* )g_malloc(yres*sizeof(gfloat));
    }
    if (nullme) PGFrectFill(dc, 0);
    return(dc); 
}

PgFRect* PGFrectNewAlike(PgFRect *frect, gboolean nullme)
{
    return PGFrectNew(frect->xres, frect->yres, frect->xreal, frect->yreal, nullme);
}

void
PGFrectFree(PgFRect *frect)
{
    gint i;
    for (i = 0; i < frect->xres; i++)
    {
        g_free((void *) frect->data[i]);
    }

    g_free((void *)frect->data);
    frect->data=NULL;
}

void
PGFrectFill(PgFRect *frect, gfloat value)
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

void PGFrectAdd(PgFRect *frect, gfloat value)
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

void PGFrectScale(PgFRect *frect, gfloat value)
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


