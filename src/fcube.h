
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


/*  fcube.h : 
 *  floating point 3D data representation
 */

#ifndef SV_FCUBE
#define SV_FCUBE

#include <glib.h>
//#include <libprocess/gwyprocess.h>

typedef struct {
   gint xres;
   gint yres;
   gint zres;
   gdouble xreal;
   gdouble yreal;
   gdouble zreal;
   gfloat ***data;
} SvFCube;

SvFCube *sv_fcube_new(gint xres, gint yres, gint zres,
                      gdouble xreal, gdouble yreal, gdouble zreal,
                      gboolean nullme);
SvFCube *sv_fcube_new_alike   (SvFCube *fcube, gboolean nullme);
void sv_fcube_free            (SvFCube *fcube);
void sv_fcube_fill            (SvFCube *fcube, gfloat value);
void sv_fcube_add             (SvFCube *fcube, gfloat value);
void sv_fcube_multiply        (SvFCube *fcube, gfloat value);
//void sv_fcube_get_datafield   (SvFCube *fcube, GwyDataField *dfield, gint i, gint j, gint k);

#endif

/* vim: set cin et ts=4 sw=4 cino=>1s,e0,n0,f0,{0,}0,^0,\:1s,=0,g1s,h0,t0,+1s,c3,(0,u0 : */

