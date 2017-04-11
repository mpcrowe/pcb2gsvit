
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
#include "fcube.h"

SvFCube*
sv_fcube_new(gint xres, gint yres, gint zres, gdouble xreal, gdouble yreal, gdouble zreal, gboolean nullme)
{
    gint i, j;
    SvFCube *dc = (SvFCube *)g_malloc(sizeof(SvFCube));

    dc->xres = xres;
    dc->yres = yres; 
    dc->zres = zres;
    dc->xreal = xreal;
    dc->yreal = yreal;
    dc->zreal = zreal;
    dc->data= (gfloat ***) g_malloc(xres*sizeof(gfloat**));
    for (i = 0; i < xres; i++) {
        dc->data[i]= (gfloat **) g_malloc(yres*sizeof(gfloat*));
        for (j = 0; j < yres; j++)
            dc->data[i][j]= (gfloat *) g_malloc(zres*sizeof(gfloat));
    }
    if (nullme) sv_fcube_fill(dc, 0);
    return dc; 
}

SvFCube*
sv_fcube_new_alike (SvFCube *fcube, gboolean nullme)
{
    return sv_fcube_new(fcube->xres, fcube->yres, fcube->zres, fcube->xreal, fcube->yreal, fcube->zreal, nullme);
}

void
sv_fcube_free(SvFCube *fcube)
{
    gint i, j;
    for (i = 0; i < fcube->xres; i++) {
        for (j = 0; j < fcube->yres; j++)
            g_free((void **) fcube->data[i][j]);
        g_free((void *) fcube->data[i]);
    }

    g_free((void *)fcube->data);
    fcube->data=NULL;
}

void
sv_fcube_fill(SvFCube *fcube, gfloat value)
{
    gint i, j, k;

#ifndef G_OS_WIN32
#pragma omp parallel default(shared) private(j, k)
#endif
    
    for (i = 0; i < fcube->xres; i++) {
        for (j = 0; j < fcube->yres; j++) {
            for (k = 0; k < fcube->zres; k++)
                fcube->data[i][j][k] = value;
        }
    }
}

void
sv_fcube_add(SvFCube *fcube, gfloat value)
{
    gint i, j, k;
    
#ifndef G_OS_WIN32
#pragma omp parallel default(shared) private(j, k)
#endif
    
    for (i = 0; i < fcube->xres; i++) {
        for (j = 0; j < fcube->yres; j++) {
            for (k = 0; k < fcube->zres; k++)
                fcube->data[i][j][k] += value;
        }
    }
}

void
sv_fcube_multiply(SvFCube *fcube, gfloat value)
{
    gint i, j, k;
    
#ifndef G_OS_WIN32
#pragma omp parallel default(shared) private(j, k)
#endif
    
    for (i = 0; i < fcube->xres; i++) {
        for (j = 0; j < fcube->yres; j++) {
            for (k = 0; k < fcube->zres; k++)
                fcube->data[i][j][k] *= value;
        }
    }
}


#if USE_GWY
void
sv_fcube_get_datafield(SvFCube *fcube, GwyDataField *dfield, gint ipos, gint jpos, gint kpos)
{
    gint i, j, k;
    if (ipos == -1 && jpos == -1) {
        gwy_data_field_resample(dfield, fcube->xres, fcube->yres, GWY_INTERPOLATION_NONE);
        for (i = 0; i < fcube->xres; i++) {
            for (j = 0; j < fcube->yres; j++)
                gwy_data_field_set_val(dfield, i, j, fcube->data[i][j][kpos]);
        }
    }
    else if (ipos == -1 && kpos == -1) {
        gwy_data_field_resample(dfield, fcube->xres, fcube->zres, GWY_INTERPOLATION_NONE);
        for (i = 0; i < fcube->xres; i++) {
            for (k = 0; k < fcube->zres; k++)
                gwy_data_field_set_val(dfield, i, k, fcube->data[i][jpos][k]);
        }
    }
    else if (jpos == -1 && kpos == -1) {
        gwy_data_field_resample(dfield, fcube->yres, fcube->zres, GWY_INTERPOLATION_NONE);
        for (j = 0; j < fcube->yres; j++) {
            for (k = 0; k < fcube->zres; k++)  
                gwy_data_field_set_val(dfield, j, k, fcube->data[ipos][j][k]);
        }
    }
}
#endif
/* vim: set cin et ts=4 sw=4 cino=>1s,e0,n0,f0,{0,}0,^0,\:1s,=0,g1s,h0,t0,+1s,c3,(0,u0 : */
